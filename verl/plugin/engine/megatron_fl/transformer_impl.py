# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from http://github.com/verl-project/verl/blob/main/verl/workers/engine/megatron/transformer_impl.py
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

"""MegatronFLEngine - Megatron Engine with FL Multi-Chip Support

This module provides support for FL (FlagOS) multi-chip devices, implementing
optimized training operations through Transformer-Engine-FL.

See docs/design/fl_multi_chip_support.md for architecture details.

Environment variables (managed by FLEnvManager):
    TE_FL_PREFER: Backend priority (flagos/vendor/reference)
    TE_FL_STRICT: Strict mode, no fallback
    TEFL_LOG_LEVEL: TE-FL log level
    USE_FLAGGEMS: FlagGems operator switch
    USE_FLAGCX: FlagCX communication switch
    TRAINING_FL_FLAGOS_WHITELIST: FlagGems operator whitelist
    TRAINING_FL_FLAGOS_BLACKLIST: FlagGems operator blacklist
"""

import logging
import os

from verl.plugin.utils import FLEnvManager, may_enable_flag_gems
from verl.trainer.config import CheckpointConfig
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron import MegatronEngineWithLMHead

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="language_model", backend="megatron", device="flagos")
class MegatronFLEngineWithLMHead(MegatronEngineWithLMHead):
    """Megatron Engine with FL (FlagOS) multi-chip support.

    This engine extends MegatronEngineWithLMHead with support for FL devices,
    leveraging Transformer-Engine-FL for optimized training operations.

    The FL environment variables should be set before this engine is initialized,
    typically by FLEnvManager in the worker initialization phase.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        # Call parent constructor
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

        logger.info("MegatronFLEngineWithLMHead initialized successfully")

    def _validate_fl_env(self):
        """Validate FL environment variables and log configuration."""
        # Log FL status summary
        logger.info(f"FL Status: {FLEnvManager.get_summary()}")

        # Check TE-FL configuration
        if not FLEnvManager.is_training_fl_enabled():
            logger.warning(
                "TE_FL_PREFER not set to 'flagos'. FL engine may not use optimized kernels. "
                "Set TE_FL_PREFER=flagos to enable FL optimizations."
            )
        else:
            training_env = FLEnvManager.get_training_env()
            logger.info(f"TE-FL configuration: {training_env}")

        # Enable FlagGems for training phase (checks is_flaggems_enabled internally)
        may_enable_flag_gems(phase="training")

        # Check FlagCX configuration
        if FLEnvManager.is_flagcx_enabled():
            logger.info(f"FlagCX communication is enabled (path: {os.environ.get('FLAGCX_PATH', 'N/A')})")

    def initialize(self):
        """Initialize the FL engine with optimized settings."""
        logger.info("Initializing MegatronFLEngineWithLMHead...")
        # Validate and log FL environment configuration
        self._validate_fl_env()
        super().initialize()
        logger.info("MegatronFLEngineWithLMHead initialization complete")
