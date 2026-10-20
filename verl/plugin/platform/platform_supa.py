# Copyright (c) 2026 BAAI. All rights reserved.
"""Biren SUPA platform implementation."""

import logging
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase

logger = logging.getLogger(__name__)


def _get_supa_module() -> ModuleType:
    """Return the ``torch.supa`` module, importing ``torch_supa`` if needed."""
    if not hasattr(torch, "supa"):
        try:
            import torch_supa  # noqa: F401 – registers torch.supa
        except (ImportError, RuntimeError, AttributeError) as err:
            raise ImportError("Biren SUPA platform requires the 'torch_supa' package.") from err
    return torch.supa


class PlatformSUPA(PlatformBase):
    """Platform backend for Biren SUPA devices."""

    @property
    def device_name(self) -> str:
        return "supa"

    @property
    def device_module(self) -> ModuleType:
        return _get_supa_module()

    def is_available(self) -> bool:
        try:
            return _get_supa_module().is_available()
        except (ImportError, AttributeError):
            return False

    def current_device(self) -> int:
        return _get_supa_module().current_device()

    def device_count(self) -> int:
        return _get_supa_module().device_count()

    def set_device(self, device_index: int) -> None:
        _get_supa_module().set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        supa = _get_supa_module()
        if device_index is None:
            supa.synchronize()
        else:
            supa.synchronize(device_index)

    def manual_seed(self, seed: int) -> None:
        _get_supa_module().manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        _get_supa_module().manual_seed_all(seed)

    def set_allocator_settings(self, settings: str) -> None:
        supa = _get_supa_module()
        memory = getattr(supa, "memory", None)
        set_settings = getattr(memory, "_set_allocator_settings", None)
        if set_settings is not None:
            set_settings(settings)

    def empty_cache(self) -> None:
        _get_supa_module().empty_cache()

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        supa = _get_supa_module()
        get_capability = getattr(supa, "get_device_capability", None)
        if get_capability is None:
            return (None, None)
        return get_capability(device_index)

    def communication_backend_name(self) -> str:
        return "bccl"

    def visible_devices_envvar(self) -> str:
        return "SUPA_VISIBLE_DEVICES"

    @contextmanager
    def nvtx_range(self, msg: str):
        logger.debug("NVTX range (no-op on SUPA): %s", msg)
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    def cudart(self) -> Any:
        return None

    def ensure_initialized(self) -> None:
        _get_supa_module()
        logger.debug("torch.supa initialised by PlatformSUPA.ensure_initialized()")
