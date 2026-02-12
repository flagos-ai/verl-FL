# Copyright 2025 FlagOS Team & Bytedance Ltd. and/or its affiliates
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

from verl.trainer.config import CheckpointConfig
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
from verl.utils.fl import FLEnvManager

from ..base import EngineRegistry
from ..megatron import MegatronEngineWithLMHead

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

    # Check if FlagGems is already imported
    if 'flag_gems' in sys.modules:
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
            raise ValueError(
                f"Cannot set both whitelist and blacklist for {phase} phase. "
                "Please set only one of them."
            )

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
            "FlagGems is not available but USE_FLAGGEMS is set. "
            "Please install FlagGems: pip install flag-gems"
        )


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
