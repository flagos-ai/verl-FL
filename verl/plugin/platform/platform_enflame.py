# Copyright (c) 2026 BAAI. All rights reserved.
"""Enflame GCU platform implementation."""

import logging
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase

logger = logging.getLogger(__name__)


def _get_gcu_module() -> ModuleType:
    """Return the ``torch.gcu`` module, importing ``torch_gcu`` if needed."""
    if not hasattr(torch, "gcu"):
        try:
            import torch_gcu  # noqa: F401 – registers torch.gcu
        except (ImportError, RuntimeError, AttributeError) as err:
            raise ImportError(
                "Enflame platform requires the 'torch_gcu' package. Please install it first."
            ) from err
    return torch.gcu


class PlatformENFLAME(PlatformBase):
    """Platform backend for Enflame GCU accelerators."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "enflame"

    @property
    def device_module(self) -> ModuleType:
        return _get_gcu_module()

    def is_available(self) -> bool:
        try:
            gcu = _get_gcu_module()
            return gcu.is_available()
        except ImportError:
            return False

    def current_device(self) -> int:
        return _get_gcu_module().current_device()

    def device_count(self) -> int:
        return _get_gcu_module().device_count()

    def set_device(self, device_index: int) -> None:
        _get_gcu_module().set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if device_index is not None:
            _get_gcu_module().synchronize(device_index)
        else:
            _get_gcu_module().synchronize()

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        _get_gcu_module().manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        _get_gcu_module().manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        gcu = _get_gcu_module()
        if hasattr(gcu, "memory") and hasattr(gcu.memory, "_set_allocator_settings"):
            gcu.memory._set_allocator_settings(settings)

    def empty_cache(self) -> None:
        _get_gcu_module().empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        gcu = _get_gcu_module()
        if hasattr(gcu, "get_device_capability"):
            return gcu.get_device_capability(device_index)
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        if os.getenv("USE_FLAGCX", "").lower() in ("1", "true"):
            return "flagcx"
        return "eccl"

    def visible_devices_envvar(self) -> str:
        return "TOPS_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        logger.debug("NVTX range (no-op on ENFLAME): %s", msg)
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        """Eagerly load ``torch_gcu`` so that downstream libraries
        (``transformers``, ``accelerate``, ``flash_attn``, …) see a fully
        initialised GCU runtime when they are imported later.

        Also patches ``torch.gcu.Stream`` to expose a ``cuda_stream`` property
        so that FlagCX's ``adaptor_stream_copy`` (which hardcodes ``cuda_stream``)
        works transparently on GCU devices.
        """
        gcu = _get_gcu_module()

        # FlagCX wrapper accesses stream.cuda_stream, but GCU streams may use
        # gcu_stream. Add a compatibility alias when needed.
        stream_cls = getattr(gcu, "Stream", None)
        if stream_cls is not None and not hasattr(stream_cls, "cuda_stream"):
            if hasattr(stream_cls, "gcu_stream"):
                stream_cls.cuda_stream = property(lambda self: self.gcu_stream)

        logger.debug("torch_gcu initialised by PlatformENFLAME.ensure_initialized()")
