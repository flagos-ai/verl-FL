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

"""FSDP FL Engine - FSDP Engine with FL Multi-Chip Support

This module provides FSDP support for FL (FlagOS) multi-chip devices,
focusing on controlling the usage of FlagGems operator library.

See docs/design/fl_multi_chip_support.md for architecture details.

Since FSDP uses PyTorch native implementation, FL support is mainly achieved
through FlagGems operator replacement:
- Control FlagGems enable/disable via environment variables
- Support operator-level whitelist/blacklist control

Environment variables (managed by FLEnvManager):
    USE_FLAGGEMS: FlagGems operator switch (true/false)
    TRAINING_FL_FLAGOS_WHITELIST: Enabled operators whitelist (comma separated)
    TRAINING_FL_FLAGOS_BLACKLIST: Disabled operators blacklist (comma separated)
    USE_FLAGCX: FlagCX communication switch
"""

import logging
import os

from verl.trainer.config import CheckpointConfig
from verl.utils.fl import FLEnvManager
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig

from ..base import EngineRegistry
from ..fsdp import FSDPEngineWithLMHead
from ..fsdp.transformer_impl import FSDPEngineWithValueHead

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def may_enable_flag_gems(phase: str = "training"):
    """Enable FlagGems operators based on FLEnvManager configuration.

    This function checks if FlagGems is available and enables it with the
    appropriate whitelist/blacklist configuration for the specified phase.

    Args:
        phase: Either "training" or "rollout". Determines which whitelist/blacklist
               environment variables to use. Defaults to "training".

    Environment Variables (for training phase):
        TRAINING_FL_FLAGOS_WHITELIST: Comma-separated list of ops to enable
        TRAINING_FL_FLAGOS_BLACKLIST: Comma-separated list of ops to disable
        TRAINING_FLAGGEMS_PATH: Path to save FlagGems records

    Environment Variables (for rollout phase):
        ROLLOUT_FL_FLAGOS_WHITELIST / VLLM_FL_FLAGOS_WHITELIST: Ops to enable
        ROLLOUT_FL_FLAGOS_BLACKLIST / VLLM_FL_FLAGOS_BLACKLIST: Ops to disable
        ROLLOUT_FLAGGEMS_PATH: Path to save FlagGems records
    """
    import sys

    # TODO: We must ensure FlagGems is only enabled once per process. This is before vllm-plugin-FL,
    # so we can't enable flag_gems here.
    return

    # Check if FlagGems is already imported
    if "flag_gems" in sys.modules:
        logger.info("FlagGems is already imported, skipping re-import")
        return

    # Check if FlagGems is enabled via FLEnvManager
    if not FLEnvManager.is_flaggems_enabled():
        logger.debug("FlagGems is not enabled (USE_FLAGGEMS not set)")
        return

    try:
        import flag_gems

        # Get whitelist and blacklist from FLEnvManager
        whitelist = FLEnvManager.get_flaggems_whitelist(phase=phase)
        blacklist = FLEnvManager.get_flaggems_blacklist(phase=phase)

        # Validate: whitelist and blacklist are mutually exclusive
        if whitelist and blacklist:
            raise ValueError(f"Cannot set both whitelist and blacklist for {phase} phase. Please set only one of them.")

        # Determine record path based on phase
        if phase == "training":
            record_path = os.environ.get("TRAINING_FLAGGEMS_PATH")
        else:
            record_path = os.environ.get("ROLLOUT_FLAGGEMS_PATH")

        # Enable FlagGems with appropriate configuration
        if whitelist:
            logger.info(f"[FlagGems][{phase}] Enable only the following ops: {whitelist}")
            flag_gems.only_enable(
                include=whitelist,
                record=True,
                once=True,
                path=record_path,
            )
        elif blacklist:
            logger.info(f"[FlagGems][{phase}] Disable the following ops: {blacklist}")
            flag_gems.enable(
                unused=blacklist,
                record=True,
                once=True,
                path=record_path,
            )
        else:
            logger.info(f"[FlagGems][{phase}] Enable all ops")
            flag_gems.enable(
                record=True,
                once=True,
                path=record_path,
            )

        logger.info(f"FlagGems version: {flag_gems.__version__}")

    except ImportError:
        logger.warning(
            "FlagGems is not available but USE_FLAGGEMS is set. Please install FlagGems: pip install flag-gems"
        )


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="flagos")
class FSDPFLEngineWithLMHead(FSDPEngineWithLMHead):
    """FL Extended FSDP Engine with LM Head

    Supports FL multi-chip device language model training with FlagGems operator acceleration.

    Environment variable control:
        USE_FLAGGEMS: Whether to enable FlagGems (true/false)
        FLAGGEMS_WHITELIST: Enabled operators whitelist
        FLAGGEMS_BLACKLIST: Disabled operators blacklist
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        # Call parent class initialization
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

        logger.info("FSDPFLEngineWithLMHead initialized successfully")

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under FSDP.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """
        logger.info(f"Initializing FSDPFLEngineWithLMHead - FL Status: {FLEnvManager.get_summary()}")

        # Enable FlagGems for training phase (checks is_flaggems_enabled internally)
        may_enable_flag_gems(phase="training")

        # Call parent initialization to build model/optimizer and setup checkpoint manager
        super().initialize()


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device="flagos")
class FSDPFLEngineWithValueHead(FSDPEngineWithValueHead):
    """FL Extended FSDP Engine with Value Head

    Supports FL multi-chip device Critic/Value model training.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        # Call parent class initialization
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

        logger.info("FSDPFLEngineWithValueHead initialized successfully")

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under FSDP.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """
        logger.info(f"Initializing FSDPFLEngineWithValueHead - FL Status: {FLEnvManager.get_summary()}")

        # Enable FlagGems for training phase (checks is_flaggems_enabled internally)
        may_enable_flag_gems(phase="training")

        # Call parent initialization to build model/optimizer and setup checkpoint manager
        super().initialize()
