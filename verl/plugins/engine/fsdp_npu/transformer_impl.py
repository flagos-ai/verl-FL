# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from https://github.com/verl-project/verl/blob/main/verl/workers/engine/fsdp/transformer_impl.py
# Below is the original copyright:

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FSDP NPU Engine - FSDP Engine with Ascend NPU Support (Example)

This module provides an example FSDP engine plugin for Huawei Ascend NPU devices.
It demonstrates how hardware vendors can extend the base FSDP engine with
device-specific optimizations through the plugin mechanism.

NPU-specific considerations:
- Uses HCCL as the distributed communication backend
- May require Ascend-specific environment variables (e.g. ASCEND_RT_VISIBLE_DEVICES)
- Supports device-specific operator configurations via torch_npu

Usage:
    This engine is auto-registered via ``EngineRegistry.register`` and selected
    when ``get_device_name()`` returns ``"npu"`` with FSDP backend.

    To use explicitly::

        from verl.workers.engine.base import EngineRegistry
        engine_cls = EngineRegistry.get_engine_cls("language_model", "fsdp")
        # On NPU hardware, this returns FSDPNPUEngineWithLMHead
"""

import logging
import os

from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithValueHead

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _maybe_configure_npu():
    """Apply Ascend NPU-specific configurations if available.

    This function is called during engine initialization to set up
    any NPU-specific optimizations or environment settings.
    """
    try:
        import torch_npu  # noqa: F401

        # Example: configure NPU-specific settings
        # torch_npu.npu.set_option(...)
        logger.debug("torch_npu is available, NPU-specific configs applied")
    except ImportError:
        logger.debug("torch_npu not available, skipping NPU-specific configuration")


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="npu")
class FSDPNPUEngineWithLMHead(FSDPEngineWithLMHead):
    """NPU Extended FSDP Engine with LM Head

    Supports Ascend NPU language model training with FSDP.

    This engine extends the base FSDP engine with NPU-specific initialization
    and optimization hooks. Hardware vendors can override methods to add
    device-specific behavior.

    Example NPU-specific features:
        - HCCL communication backend (auto-configured via platform plugin)
        - Ascend-specific memory management
        - Custom operator dispatching
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPNPUEngineWithLMHead initialized successfully")

    def initialize(self):
        """Build the model, optimizer, and LR scheduler under FSDP for NPU.

        Applies NPU-specific configurations before calling parent initialization.
        """
        logger.info("Initializing FSDPNPUEngineWithLMHead...")

        # Apply NPU-specific settings
        _maybe_configure_npu()

        # Call parent initialization to build model/optimizer
        super().initialize()
        logger.info("FSDPNPUEngineWithLMHead initialization complete")


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device="npu")
class FSDPNPUEngineWithValueHead(FSDPEngineWithValueHead):
    """NPU Extended FSDP Engine with Value Head

    Supports Ascend NPU Critic/Value model training with FSDP.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPNPUEngineWithValueHead initialized successfully")

    def initialize(self):
        """Build the model, optimizer, and LR scheduler under FSDP for NPU."""
        logger.info("Initializing FSDPNPUEngineWithValueHead...")

        _maybe_configure_npu()

        super().initialize()
        logger.info("FSDPNPUEngineWithValueHead initialization complete")
