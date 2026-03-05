# Copyright (c) 2026 BAAI. All rights reserved.

"""Huawei Ascend NPU platform implementation."""

from __future__ import annotations

import torch

from .base import BasePlatform, PlatformInfo


class NPUPlatform(BasePlatform):
    """Huawei Ascend NPU platform."""

    info = PlatformInfo(
        vendor_name="ascend",
        device_name="npu",
        device_query_cmd="npu-smi info",
        comm_backend="hccl",
        visible_devices_env="ASCEND_RT_VISIBLE_DEVICES",
        dispatch_key="PrivateUse1",
    )

    def is_available(self) -> bool:
        try:
            if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
                return torch.npu.is_available()
            return False
        except Exception:
            return False
