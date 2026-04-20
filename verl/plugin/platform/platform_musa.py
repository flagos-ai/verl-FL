# Copyright (c) 2026 BAAI. All rights reserved.
"""Moore Threads MUSA platform implementation."""

import logging
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase

logger = logging.getLogger(__name__)


def _get_musa_module() -> ModuleType:
    """Return the ``torch.musa`` module, importing ``torch_musa`` if needed."""
    if not hasattr(torch, "musa"):
        try:
            import torch_musa  # noqa: F401 – registers torch.musa
        except (ImportError, RuntimeError, AttributeError) as err:
            raise ImportError(
                "Moore Threads MUSA platform requires the 'torch_musa' package. Please install it first."
            ) from err
    return torch.musa


class PlatformMUSA(PlatformBase):
    """Platform backend for Moore Threads MUSA GPUs."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "musa"

    @property
    def device_module(self) -> ModuleType:
        return _get_musa_module()

    def is_available(self) -> bool:
        try:
            musa = _get_musa_module()
            return musa.is_available()
        except ImportError:
            return False

    def current_device(self) -> int:
        return _get_musa_module().current_device()

    def device_count(self) -> int:
        return _get_musa_module().device_count()

    def set_device(self, device_index: int) -> None:
        _get_musa_module().set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if device_index is not None:
            _get_musa_module().synchronize(device_index)
        else:
            _get_musa_module().synchronize()

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        _get_musa_module().manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        _get_musa_module().manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        musa = _get_musa_module()
        if hasattr(musa, "memory") and hasattr(musa.memory, "_set_allocator_settings"):
            musa.memory._set_allocator_settings(settings)

    def empty_cache(self) -> None:
        _get_musa_module().empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        musa = _get_musa_module()
        if hasattr(musa, "get_device_capability"):
            return musa.get_device_capability(device_index)
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        if os.getenv("USE_FLAGCX", "0").lower() in ["1", "true"]:
            return "flagcx"
        # mccl is the Moore Threads collective communication library
        return "mccl"

    def visible_devices_envvar(self) -> str:
        return "MUSA_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        logger.debug("NVTX range (no-op on MUSA): %s", msg)
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
        """Eagerly load ``torch_musa`` so that downstream libraries
        (``transformers``, ``accelerate``, ``flash_attn``, …) see a fully
        initialised MUSA runtime when they are imported later."""
        _get_musa_module()
        logger.debug("torch_musa initialised by PlatformMUSA.ensure_initialized()")
