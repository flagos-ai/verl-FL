# Copyright (c) 2026 BAAI. All rights reserved.

"""FL Environment Manager for verl.

This module provides a lightweight environment variable manager for FL (FlagOS)
multi-chip support, allowing separate control of training and rollout phases.

Environment Variables:
----------------------
Training Phase (TE-FL / Megatron / FSDP):
    TE_FL_PREFER: Backend priority (flagos/vendor/reference)
    TE_FL_STRICT: Strict mode, no fallback (1/0)
    TE_FL_ALLOW_VENDORS: Allowed vendors whitelist (nvidia,amd)
    TE_FL_DENY_VENDORS: Denied vendors blacklist
    TE_FL_PER_OP: Per-op configuration (rmsnorm_fwd=vendor:cuda|default)
    TEFL_LOG_LEVEL: Log level (DEBUG/INFO/WARNING/ERROR)
    TRAINING_FL_FLAGOS_WHITELIST: FlagGems operator whitelist for training
    TRAINING_FL_FLAGOS_BLACKLIST: FlagGems operator blacklist for training

Rollout Phase (vLLM):
    VLLM_FL_PREFER_ENABLED: Enable FL preference (true/false)
    VLLM_FL_PLATFORM: Platform type (cuda/npu)
    VLLM_FL_PREFER: Backend priority (flagos/vendor)
    VLLM_FL_OOT_ENABLED: Enable out-of-tree plugins (1/0)
    VLLM_FL_FLAGOS_WHITELIST: FlagGems operator whitelist for rollout
    VLLM_FL_FLAGOS_BLACKLIST: FlagGems operator blacklist for rollout

Common:
    USE_FLAGGEMS: Enable FlagGems globally (true/false/1/0)
    USE_FLAGCX: Enable FlagCX communication (1/0)
    FLAGCX_PATH: Path to FlagCX installation
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class FLEnvManager:
    """Lightweight FL environment variable manager.

    This class provides static methods to manage FL environment variables
    for training and rollout phases separately.
    """

    # Training phase environment variable keys
    TRAINING_ENV_KEYS = [
        # TE-FL configuration
        "TE_FL_PREFER",
        "TE_FL_STRICT",
        "TE_FL_ALLOW_VENDORS",
        "TE_FL_DENY_VENDORS",
        "TE_FL_PER_OP",
        "TE_FL_PREFER_VENDOR",
        "TEFL_LOG_LEVEL",
        # Training FlagGems configuration
        "TRAINING_FL_FLAGOS_WHITELIST",
        "TRAINING_FL_FLAGOS_BLACKLIST",
    ]

    # Rollout phase environment variable keys
    ROLLOUT_ENV_KEYS = [
        # vLLM-FL configuration
        "VLLM_FL_PREFER_ENABLED",
        "VLLM_FL_PLATFORM",
        "VLLM_FL_PREFER",
        "VLLM_FL_OOT_ENABLED",
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_CONFIG",  # the custom config for vLLM-FL
    ]

    # Common environment variable keys
    COMMON_ENV_KEYS = [
        "USE_FLAGGEMS",
        "USE_FLAGCX",
        "FLAGCX_PATH",
    ]

    # Internal state for context management
    _saved_env: dict[str, Optional[str]] = {}
    _current_phase: Optional[str] = None

    @classmethod
    def is_fl_enabled(cls) -> bool:
        """Check if FL multi-chip support is enabled.

        FL is considered enabled if TE_FL_PREFER is set to 'flagos'
        or VLLM_FL_PREFER is set to 'flagos'.

        Returns:
            bool: True if FL is enabled, False otherwise.
        """
        te_fl_prefer = os.environ.get("TE_FL_PREFER", "").lower()
        vllm_fl_prefer = os.environ.get("VLLM_FL_PREFER", "").lower()
        return te_fl_prefer == "flagos" or vllm_fl_prefer == "flagos"

    @classmethod
    def is_training_fl_enabled(cls) -> bool:
        """Check if FL is enabled for training phase.

        Returns:
            bool: True if training FL is enabled.
        """
        return os.environ.get("TE_FL_PREFER", "").lower() == "flagos"

    @classmethod
    def is_rollout_fl_enabled(cls) -> bool:
        """Check if FL is enabled for rollout phase.

        Returns:
            bool: True if rollout FL is enabled.
        """
        return os.environ.get("VLLM_FL_PREFER", "").lower() == "flagos"

    @classmethod
    def is_flaggems_enabled(cls) -> bool:
        """Check if FlagGems is enabled.

        Returns:
            bool: True if FlagGems is enabled.
        """
        use_flaggems = os.environ.get("USE_FLAGGEMS", "").lower()
        return use_flaggems in ("true", "1", "yes")

    @classmethod
    def is_flagcx_enabled(cls) -> bool:
        """Check if FlagCX is enabled.

        Returns:
            bool: True if FlagCX is enabled.
        """
        if os.environ.get("USE_FLAGCX", "").lower() in ("true", "1"):
            if os.environ.get("FLAGCX_PATH") is None:
                raise RuntimeError("USE_FLAGCX is set but FLAGCX_PATH is not defined")
            return True
        return False

    @classmethod
    def get_training_env(cls) -> dict[str, str]:
        """Get all training phase FL environment variables.

        Returns:
            Dict[str, str]: Dictionary of training environment variables.
        """
        env = {}
        for key in cls.TRAINING_ENV_KEYS + cls.COMMON_ENV_KEYS:
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
        return env

    @classmethod
    def get_rollout_env(cls) -> dict[str, str]:
        """Get all rollout phase FL environment variables.

        Returns:
            Dict[str, str]: Dictionary of rollout environment variables.
        """
        env = {}
        for key in cls.ROLLOUT_ENV_KEYS + cls.COMMON_ENV_KEYS:
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
        return env

    @classmethod
    def get_flaggems_whitelist(cls, phase: str = "training") -> Optional[list]:
        """Get FlagGems operator whitelist for specified phase.

        Args:
            phase: Either "training" or "rollout".

        Returns:
            Optional[list]: List of whitelisted operators, or None if not set.
        """
        if phase == "training":
            whitelist_str = os.environ.get("TRAINING_FL_FLAGOS_WHITELIST", "")
        else:
            whitelist_str = os.environ.get("VLLM_FL_FLAGOS_WHITELIST", "")

        if whitelist_str:
            return [op.strip() for op in whitelist_str.split(",") if op.strip()]
        return None

    @classmethod
    def get_flaggems_blacklist(cls, phase: str = "training") -> Optional[list]:
        """Get FlagGems operator blacklist for specified phase.

        Args:
            phase: Either "training" or "rollout".

        Returns:
            Optional[list]: List of blacklisted operators, or None if not set.
        """
        if phase == "training":
            blacklist_str = os.environ.get("TRAINING_FL_FLAGOS_BLACKLIST", "")
        else:
            blacklist_str = os.environ.get("VLLM_FL_FLAGOS_BLACKLIST", "")

        if blacklist_str:
            return [op.strip() for op in blacklist_str.split(",") if op.strip()]
        return None

    @classmethod
    def get_current_phase(cls) -> Optional[str]:
        """Get the current phase if within a context.

        Returns:
            Optional[str]: "training", "rollout", or None if not in a context.
        """
        return cls._current_phase

    @classmethod
    def log_fl_status(cls):
        """Log the current FL configuration status."""
        logger.info("=" * 60)
        logger.info("FL Multi-Chip Support Status")
        logger.info("=" * 60)
        logger.info(f"FL Enabled: {cls.is_fl_enabled()}")
        logger.info(f"Training FL Enabled: {cls.is_training_fl_enabled()}")
        logger.info(f"Rollout FL Enabled: {cls.is_rollout_fl_enabled()}")
        logger.info(f"FlagGems Enabled: {cls.is_flaggems_enabled()}")
        logger.info(f"FlagCX Enabled: {cls.is_flagcx_enabled()}")
        logger.info("-" * 60)
        logger.info("Training Environment:")
        for key, value in cls.get_training_env().items():
            logger.info(f"  {key}={value}")
        logger.info("-" * 60)
        logger.info("Rollout Environment:")
        for key, value in cls.get_rollout_env().items():
            logger.info(f"  {key}={value}")
        logger.info("=" * 60)

    @classmethod
    def get_summary(cls) -> str:
        """Get a summary string of FL configuration.

        Returns:
            str: Summary of FL configuration.
        """
        parts = []
        if cls.is_training_fl_enabled():
            parts.append(f"training(TE_FL={os.environ.get('TE_FL_PREFER', 'N/A')})")
        if cls.is_rollout_fl_enabled():
            parts.append(f"rollout(VLLM_FL={os.environ.get('VLLM_FL_PREFER', 'N/A')})")
        if cls.is_flaggems_enabled():
            parts.append("FlagGems")
        if cls.is_flagcx_enabled():
            parts.append("FlagCX")

        if parts:
            return f"FL[{', '.join(parts)}]"
        return "FL[disabled]"


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

    # FlagGems initialization is handled by vllm-plugin-FL during rollout setup.
    # Since verl uses colocated processes for training and rollout, enabling FlagGems
    # here would cause duplicate initialization. This will be revisited when verl
    # supports separate processes for training and rollout phases.
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
