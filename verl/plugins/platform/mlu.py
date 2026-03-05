# Copyright (c) 2026 BAAI. All rights reserved.

"""Cambricon MLU platform implementation."""

from __future__ import annotations

import importlib.util

import torch

from .base import BasePlatform, PlatformInfo


class MLUPlatform(BasePlatform):
    """Cambricon MLU platform."""

    info = PlatformInfo(
        vendor_name="cambricon",
        device_name="mlu",
        device_query_cmd="cnmon",
        comm_backend="cncl",
        visible_devices_env="MLU_VISIBLE_DEVICES",
        dispatch_key="PrivateUse1",
        support_fp64=False,
    )

    def is_available(self) -> bool:
        try:
            if importlib.util.find_spec("torch_mlu") is None:
                return False
            import torch_mlu  # noqa: F401

            return hasattr(torch, "mlu") and torch.mlu.is_available()
        except Exception:
            return False
