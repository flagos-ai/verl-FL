# Copyright (c) 2026 BAAI. All rights reserved.

"""NVIDIA CUDA platform implementation (default)."""

from __future__ import annotations

from typing import Optional

import torch

from .base import BasePlatform, PlatformInfo


class CUDAPlatform(BasePlatform):
    """NVIDIA CUDA platform."""

    info = PlatformInfo(
        vendor_name="nvidia",
        device_name="cuda",
        device_query_cmd="nvidia-smi",
        comm_backend="nccl",
        visible_devices_env="CUDA_VISIBLE_DEVICES",
        dispatch_key="CUDA",
    )

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def get_device_capability(self, device_id: int = 0) -> tuple[Optional[int], Optional[int]]:
        if not self.is_available():
            return (None, None)
        return torch.cuda.get_device_capability(device_id)

    def set_expandable_segments(self, enable: bool) -> None:
        if self.is_available():
            torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
