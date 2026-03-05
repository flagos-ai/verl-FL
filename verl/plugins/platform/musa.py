# Copyright (c) 2026 BAAI. All rights reserved.

"""Moore Threads MUSA platform implementation."""

from __future__ import annotations

import importlib.util

import torch

from .base import BasePlatform, PlatformInfo


class MUSAPlatform(BasePlatform):
    """Moore Threads MUSA platform."""

    info = PlatformInfo(
        vendor_name="mthreads",
        device_name="musa",
        device_query_cmd="mthreads-gmi",
        comm_backend="mccl",
        visible_devices_env="MUSA_VISIBLE_DEVICES",
        dispatch_key="PrivateUse1",
    )

    def is_available(self) -> bool:
        try:
            if importlib.util.find_spec("torch_musa") is None:
                return False
            import torch_musa  # noqa: F401

            return hasattr(torch, "musa") and torch.musa.is_available()
        except Exception:
            return False
