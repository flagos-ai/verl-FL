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
        if os.getenv("USE_FLAGCX", "").lower() in ("1", "true"):
            return "flagcx"
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

    @staticmethod
    def _patch_flagcx_stream() -> None:
        """Expose a ``cuda_stream`` property on ``torch.musa.Stream``.

        FlagCX's ``adaptor_stream_copy`` hardcodes ``stream.cuda_stream``,
        but MUSA streams use ``musa_stream``.  This alias keeps FlagCX
        working without modifying its upstream sources.
        """
        musa = _get_musa_module()
        stream_cls = getattr(musa, "Stream", None)
        if stream_cls is not None and not hasattr(stream_cls, "cuda_stream"):
            stream_cls.cuda_stream = property(lambda self: self.musa_stream)
            logger.debug("FlagCX stream.cuda_stream alias applied")

    @staticmethod
    def _patch_module_cuda() -> None:
        """Redirect ``torch.nn.Module.cuda()`` and ``torch.Tensor.cuda()`` to the
        MUSA device.

        Third-party libraries like ``mbridge`` hardcode
        ``model.cuda(torch.cuda.current_device())``.  We intercept the ``.cuda()``
        method itself and redirect it to ``self.to("musa:...")``.

        ``torch.cuda`` module is **not** patched here, so
        ``torch.cuda.is_available()`` stays truthful on this platform and does not
        trick Megatron-LM-FL's platform auto-detection.
        """
        musa = _get_musa_module()

        def _musa_module_cuda(self, device=None, **kwargs):
            if device is None:
                device = musa.current_device()
            if isinstance(device, int):
                device = torch.device("musa", device)
            return self.to(device, **{k: v for k, v in kwargs.items() if k != "non_blocking"})

        if hasattr(torch.nn.Module, "cuda"):
            torch.nn.Module.cuda = _musa_module_cuda
        if hasattr(torch.Tensor, "cuda"):
            torch.Tensor.cuda = _musa_module_cuda
        logger.debug("Module.cuda / Tensor.cuda redirected to musa")

    def ensure_initialized(self) -> None:
        """Eagerly load ``torch_musa`` so that downstream libraries
        (``transformers``, ``accelerate``, ``flash_attn``, …) see a fully
        initialised MUSA runtime when they are imported later.

        Applies vendor-compatibility patches for third-party libraries that
        hardcode CUDA-specific APIs.
        """
        _get_musa_module() # ensure torch_musa is loaded

        PlatformMUSA._patch_flagcx_stream()
        PlatformMUSA._patch_module_cuda()

        logger.debug("torch_musa initialised by PlatformMUSA.ensure_initialized()")
