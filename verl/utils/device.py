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
import os
import platform
import subprocess

import torch
from packaging import version

from verl.plugin.platform import get_platform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level availability flags (kept for backward compatibility)
# ---------------------------------------------------------------------------


def is_torch_npu_available(check_device=True) -> bool:
    """Check if Ascend NPU is available for PyTorch operations.

    Attempts to detect NPU availability by checking for the torch.npu module
    and its is_available() function.

    Args:
        check_device : only check torch_npu package or strictly check if NPU device is available

    Returns:
        bool: True if NPU is available, False otherwise.
    """
    try:
        if not hasattr(torch, "npu"):
            return False

        if check_device:
            return torch.npu.is_available()
        else:
            return True
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


# ---------------------------------------------------------------------------
# Device info helpers
# ---------------------------------------------------------------------------


def get_resource_name() -> str:
    """Function that return ray resource name based on the device type.
    Returns:
        ray resource name string, either "GPU" or "NPU".
    """
    return "GPU" if is_cuda_available else "NPU"


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


# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------


def manual_seed(seed: int) -> None:
    """Set the seed for the current accelerator device."""
    get_platform().manual_seed(seed)


def manual_seed_all(seed: int) -> None:
    """Set the seed for all accelerator devices."""
    get_platform().manual_seed_all(seed)


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

    Returns:
        tuple: (major, minor) version numbers, or (None, None) if not applicable.
    """
    return get_platform().get_device_capability(device_id)


def is_device_available() -> bool:
    """Check if any accelerator device is available.

    Returns:
        bool: True if any accelerator is available.
    """
    return get_platform().is_available()


# ---------------------------------------------------------------------------
# NPU version / IPC support (from upstream v0.7.1)
# ---------------------------------------------------------------------------


def get_npu_versions() -> tuple:
    """Get the NPU software version and CANN toolkit version.

    Returns:
        tuple: (software_version, cann_version) strings.

    Raises:
        RuntimeError: If npu-smi command fails or versions cannot be determined.
    """
    system = platform.system()
    if system == "Linux":
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True)
        output = result.stdout

        software_version = None
        for line in output.split("\n"):
            if "Software Version" in line:
                software_version = line.split(":")[-1].strip()
                break

        if software_version is None:
            raise RuntimeError("Could not find Software Version in npu-smi output")

        cann_path = os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
        version_file = os.path.join(cann_path, "version.cfg")

        if not os.path.exists(version_file):
            raise RuntimeError(f"CANN version file not found at {version_file}")

        with open(version_file) as f:
            for line in f:
                if "CANN_VERSION" in line:
                    cann_version = line.split("=")[-1].strip()
                    return software_version, cann_version

        raise RuntimeError("Could not find CANN_VERSION in version.cfg")
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def check_ipc_version_support(software_version: str, cann_version: str) -> bool:
    """Check if the given software and CANN versions support IPC.

    IPC is supported when:
    - Software version >= 25.3.RC1 AND CANN version >= 8.3.RC1

    Args:
        software_version: The NPU software version string (e.g., "25.3.RC1")
        cann_version: The CANN toolkit version string (e.g., "8.3.RC1")

    Returns:
        bool: True if IPC is supported, False otherwise.
    """
    # Normalize version strings for comparison
    software_base = software_version.lower().replace("rc", ".rc")
    cann_base = cann_version.lower().replace("rc", ".rc")

    if version.parse(software_base) >= version.parse("25.3.rc1"):
        if version.parse(cann_base) >= version.parse("8.3.rc1"):
            return True
        else:
            logger.info(f"CANN version {cann_version} is below 8.3.RC1")
    else:
        logger.info(f"Software version {software_version} is below 25.3.rc1")

    return False


def is_support_ipc() -> bool:
    """Check if the device supports IPC (Inter-Process Communication).

    For GPU devices, always returns True.
    For NPU devices, checks the software version and CANN toolkit version
    to determine if IPC is supported.

    Returns:
        bool: True if IPC is supported, False otherwise.
    """
    # If CUDA is available, it's a GPU device
    if is_cuda_available:
        return True

    # For NPU devices, check the software version and CANN toolkit version
    if is_npu_available:
        try:
            software_version, cann_version = get_npu_versions()
            return check_ipc_version_support(software_version, cann_version)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to execute npu-smi command: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error checking IPC support: {e}") from e

    # For other devices (CPU), return False
    return False
