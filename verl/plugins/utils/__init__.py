# Copyright (c) 2026 BAAI. All rights reserved.

"""FL (FlagOS) Multi-Chip Support Utilities for verl.

This module provides utilities for managing FL environment variables
across different phases (training and rollout) in the verl framework.
"""

from .config_manager import FLEnvManager, may_enable_flag_gems

__all__ = ["FLEnvManager", "may_enable_flag_gems"]
