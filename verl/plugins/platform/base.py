# Copyright (c) 2026 BAAI. All rights reserved.

"""Platform abstraction base classes for verl multi-chip support.

Defines PlatformInfo (hardware metadata) and BasePlatform (device interface)
that each hardware vendor implements. Inspired by FlagGems VendorInfoBase
and HF Accelerate device detection patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class PlatformInfo:
    """Hardware metadata that each vendor fills in.

    Attributes:
        vendor_name: Vendor identifier, e.g. "nvidia", "cambricon", "ascend".
        device_name: PyTorch device type, e.g. "cuda", "mlu", "npu".
            Corresponds to ``torch.{device_name}``.
        device_query_cmd: Shell command to query device status,
            e.g. "nvidia-smi", "cnmon", "npu-smi info".
        comm_backend: Distributed communication backend,
            e.g. "nccl", "cncl", "hccl".
        visible_devices_env: Environment variable for visible device selection,
            e.g. "CUDA_VISIBLE_DEVICES", "MLU_VISIBLE_DEVICES".
        dispatch_key: PyTorch dispatch key, e.g. "CUDA", "PrivateUse1".
            None means same as device_name upper-cased.
        support_fp64: Whether the device supports FP64 computation.
        support_bf16: Whether the device supports BF16 computation.
    """

    vendor_name: str
    device_name: str
    device_query_cmd: str
    comm_backend: str
    visible_devices_env: str
    dispatch_key: Optional[str] = None
    support_fp64: bool = True
    support_bf16: bool = True


class BasePlatform:
    """Abstract base class for hardware platform implementations.

    Most methods have a default implementation that dynamically dispatches
    to ``getattr(torch, self.info.device_name)``, so vendors only need to
    override methods where their hardware differs.
    """

    info: PlatformInfo

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if this platform's hardware is detected."""
        mod = self._get_module(silent=True)
        if mod is None:
            return False
        return getattr(mod, "is_available", lambda: False)()

    # ------------------------------------------------------------------
    # Torch device module access
    # ------------------------------------------------------------------

    def get_torch_device_module(self) -> Any:
        """Return the ``torch.<device_name>`` module (e.g. ``torch.cuda``)."""
        return self._get_module()

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def current_device_id(self) -> int:
        return self.get_torch_device_module().current_device()

    def device_count(self) -> int:
        return self.get_torch_device_module().device_count()

    def set_device(self, device_id: int) -> None:
        self.get_torch_device_module().set_device(device_id)

    # ------------------------------------------------------------------
    # Random seed
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        self.get_torch_device_module().manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        self.get_torch_device_module().manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def empty_cache(self) -> None:
        self.get_torch_device_module().empty_cache()

    def synchronize(self) -> None:
        self.get_torch_device_module().synchronize()

    def memory_allocated(self, device=None) -> int:
        return self.get_torch_device_module().memory_allocated(device)

    def max_memory_allocated(self, device=None) -> int:
        return self.get_torch_device_module().max_memory_allocated(device)

    # ------------------------------------------------------------------
    # Device capability
    # ------------------------------------------------------------------

    def get_device_capability(self, device_id: int = 0) -> tuple[Optional[int], Optional[int]]:
        """Return ``(major, minor)`` compute capability, or ``(None, None)``."""
        return (None, None)

    # ------------------------------------------------------------------
    # Allocator settings
    # ------------------------------------------------------------------

    def set_expandable_segments(self, enable: bool) -> None:
        """Configure expandable segments for the memory allocator.

        Default is no-op; CUDA overrides this.
        """
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_module(self, silent: bool = False) -> Any:
        """Get ``torch.<device_name>`` module, with optional silent failure."""
        mod = getattr(torch, self.info.device_name, None)
        if mod is None and not silent:
            raise AttributeError(
                f"torch.{self.info.device_name} is not available. "
                f"Make sure the {self.info.vendor_name} backend is installed."
            )
        return mod
