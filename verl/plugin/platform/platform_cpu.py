# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU-only fallback platform implementation."""

import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase


class PlatformCPU(PlatformBase):
    """Fallback platform backend when no accelerator is available."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "cpu"

    @property
    def device_module(self) -> ModuleType:
        return torch.cpu

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    def device_count(self) -> int:
        # Prioritize LOCAL_SIZE environment variable if set
        if "LOCAL_SIZE" in os.environ:
            return int(os.environ.get("LOCAL_SIZE"))

        # Fallback to actual CPU count, minimum of 1
        cpu_count = os.cpu_count()
        return cpu_count if cpu_count is not None else 1

    def set_device(self, device_index: int) -> None:
        pass

    def synchronize(self, device_index: Optional[int] = None) -> None:
        pass

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        pass

    def empty_cache(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        return "gloo"

    def visible_devices_envvar(self) -> str:
        return "CUDA_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
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
