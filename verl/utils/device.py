# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

"""Backward-compatible thin wrapper.

New code should use ``verl.plugins.platform`` directly.  All public names
previously defined here are now re-exported from the platform plugin system
so that existing ``from verl.utils.device import ...`` call sites continue
to work without modification.
"""

from verl.plugins.platform import (  # noqa: F401
    auto_set_device,
    get_device_capability,
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    get_visible_devices_keyword,
    is_cuda_available,
    is_npu_available,
    set_expandable_segments,
)

# Preserve the old module-level boolean attributes that some callers reference
# directly (e.g. ``from verl.utils.device import is_npu_available``
# then ``if is_npu_available:``).  The functions above return booleans when
# called, but legacy code may use them as bare values.
# We keep them as lazy calls here so the detection runs once on first access.
is_cuda_available = is_cuda_available()  # type: ignore[assignment]
is_npu_available = is_npu_available()  # type: ignore[assignment]
