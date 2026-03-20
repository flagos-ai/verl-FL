# Copyright (c) 2026 BAAI. All rights reserved.

"""verl engine plugins.

This package contains engine implementations for various hardware backends
that extend the base engines in ``verl.workers.engine``.

Supported plugins:
    - ``fsdp_fl``:   FSDP engine for FL (FlagOS) multi-chip devices
    - ``megatron_fl``: Megatron engine for FL (FlagOS) multi-chip devices
    - ``fsdp_npu``:  FSDP engine for Ascend NPU devices (example)
"""
