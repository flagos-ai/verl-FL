# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

"""Backward-compatible device utilities.

All public names in this module are preserved for existing callers (80+ import
sites).  Internally every function now delegates to the platform abstraction
layer in :mod:`verl.plugin.platform`.
"""

import logging

import torch

from verl.plugin.platform import get_platform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level availability flags (kept for backward compatibility)
# ---------------------------------------------------------------------------


def is_torch_npu_available() -> bool:
    """Check if Ascend NPU is available for PyTorch operations."""
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
            return torch.npu.is_available()
        return False
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


def is_torch_musa_available(check_device=True) -> bool:
    """Check if Moore Threads MUSA is available for PyTorch operations.

    Checks ``sys.modules`` instead of ``hasattr(torch, "musa")`` to avoid
    triggering ``torch.__getattr__("musa")`` which would attempt
    ``import torch_musa`` and can leave a half-initialised module in
    ``sys.modules``, breaking later imports (e.g. from flash_attn).

    Args:
        check_device : only check torch_musa package or strictly check if MUSA device is available

    Returns:
        bool: True if MUSA is available, False otherwise.
    """
    try:
        import sys

        if "torch_musa" not in sys.modules:
            return False

        if not hasattr(torch, "musa"):
            return False

        if check_device:
            return torch.musa.is_available()
        else:
            return True
    except Exception:
        return False


is_musa_available = is_torch_musa_available()


# ---------------------------------------------------------------------------
# Device info helpers
# ---------------------------------------------------------------------------


def get_visible_devices_keyword() -> str:
    """Get the environment variable name for visible device selection.

    Returns:
        str: e.g. 'CUDA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES'.
    """
    return get_platform().visible_devices_envvar()


def get_device_name() -> str:
    """Get the device type string based on available accelerators.

    Returns:
        str: Device type string ('cuda', 'npu', 'cpu', …).
    """
    return get_platform().device_name


def get_torch_device():
    """Get the PyTorch device module for the current accelerator.

    Returns:
        module: The PyTorch device module (torch.cuda, torch.npu, etc.).
    """
    return get_platform().device_module


def get_device_id() -> int:
    """Get the index of the current accelerator device.

    Returns:
        int: The current device index (e.g., 0 for 'cuda:0').
    """
    return get_platform().current_device()


def get_nccl_backend() -> str:
    """Get the distributed communication backend based on device type.

    Returns:
        str: Backend name ('nccl', 'hccl', 'gloo', …).
    """
    return get_platform().communication_backend_name()


def get_dist_backend() -> str:
    """Get the compound backend string for ``init_process_group``.

    Returns ``"gloo"`` on CPU, otherwise ``"cpu:gloo,<device>:<comm>"``
    (e.g. ``"cpu:gloo,cuda:nccl"``, ``"cpu:gloo,musa:flagcx"``).
    Works for any device type — no device-specific branches needed.
    """
    device_name = get_device_name()
    if device_name == "cpu":
        return "gloo"
    return f"cpu:gloo,{device_name}:{get_nccl_backend()}"


# ---------------------------------------------------------------------------
# Memory / allocator
# ---------------------------------------------------------------------------


def set_expandable_segments(enable: bool) -> None:
    """Configure memory allocator expandable segments setting.

    Args:
        enable: If True, enable expandable segments. If False, disable them.
    """
    get_platform().set_allocator_settings(f"expandable_segments:{enable}")


# ---------------------------------------------------------------------------
# Device auto-configuration
# ---------------------------------------------------------------------------


def auto_set_device(config) -> None:
    """Automatically configure device name for different accelerators.

    Args:
        config: Configuration object with trainer.device attribute.
    """
    if config and hasattr(config, "trainer") and hasattr(config.trainer, "device"):
        detected = get_platform().device_name
        # Only override when the config value doesn't match the detected platform
        if detected != "cpu" and config.trainer.device != detected:
            if config.trainer.device != "cpu":
                logger.warning(
                    f"Detect setting config.trainer.device to {config.trainer.device} for "
                    f"{detected}, automatically set to `{detected}` instead."
                )
            config.trainer.device = detected


# ---------------------------------------------------------------------------
# Device properties
# ---------------------------------------------------------------------------


def get_device_capability(device_id: int = 0) -> tuple[int | None, int | None]:
    """Get the compute capability of the current accelerator device.

    Args:
        device_id: The device index to query. Defaults to 0.

    Returns:
        tuple: (major, minor) or (None, None) if not applicable.
    """
    return get_platform().get_device_capability(device_id)


def is_device_available() -> bool:
    """Check if any accelerator device is available.

    Returns:
        bool: True if any accelerator is available.
    """
    return get_platform().is_available()


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------


def manual_seed(seed: int) -> None:
    """Set the seed for the current accelerator device.

    Args:
        seed: The desired seed.
    """
    get_platform().manual_seed(seed)


def manual_seed_all(seed: int) -> None:
    """Set the seed for all accelerator devices.

    Args:
        seed: The desired seed.
    """
    get_platform().manual_seed_all(seed)
