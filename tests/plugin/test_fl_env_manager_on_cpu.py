# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for FLEnvManager and may_enable_flag_gems."""

import os
from unittest.mock import patch

import pytest

from verl.plugin.utils.config_manager import FLEnvManager, may_enable_flag_gems


class TestFLEnvManagerIsEnabled:
    """Tests for FL enabled checks."""

    def test_is_fl_enabled_with_te_fl_flagos(self):
        with patch.dict(os.environ, {"TE_FL_PREFER": "flagos"}, clear=False):
            assert FLEnvManager.is_fl_enabled() is True

    def test_is_fl_enabled_with_vllm_fl_flagos(self):
        with patch.dict(os.environ, {"VLLM_FL_PREFER": "flagos"}, clear=False):
            assert FLEnvManager.is_fl_enabled() is True

    def test_is_fl_enabled_with_both_flagos(self):
        with patch.dict(os.environ, {"TE_FL_PREFER": "flagos", "VLLM_FL_PREFER": "flagos"}, clear=False):
            assert FLEnvManager.is_fl_enabled() is True

    def test_is_fl_disabled_by_default(self):
        env = {k: v for k, v in os.environ.items() if k not in ("TE_FL_PREFER", "VLLM_FL_PREFER")}
        with patch.dict(os.environ, env, clear=True):
            assert FLEnvManager.is_fl_enabled() is False

    def test_is_fl_disabled_with_vendor(self):
        with patch.dict(os.environ, {"TE_FL_PREFER": "vendor", "VLLM_FL_PREFER": "vendor"}, clear=False):
            assert FLEnvManager.is_fl_enabled() is False

    def test_is_fl_enabled_case_insensitive(self):
        with patch.dict(os.environ, {"TE_FL_PREFER": "FlagOS"}, clear=False):
            assert FLEnvManager.is_fl_enabled() is True


class TestFLEnvManagerTrainingEnabled:
    """Tests for training phase FL check."""

    def test_training_fl_enabled(self):
        with patch.dict(os.environ, {"TE_FL_PREFER": "flagos"}, clear=False):
            assert FLEnvManager.is_training_fl_enabled() is True

    def test_training_fl_disabled(self):
        env = {k: v for k, v in os.environ.items() if k != "TE_FL_PREFER"}
        with patch.dict(os.environ, env, clear=True):
            assert FLEnvManager.is_training_fl_enabled() is False

    def test_training_fl_disabled_with_reference(self):
        with patch.dict(os.environ, {"TE_FL_PREFER": "reference"}, clear=False):
            assert FLEnvManager.is_training_fl_enabled() is False


class TestFLEnvManagerRolloutEnabled:
    """Tests for rollout phase FL check."""

    def test_rollout_fl_enabled(self):
        with patch.dict(os.environ, {"VLLM_FL_PREFER": "flagos"}, clear=False):
            assert FLEnvManager.is_rollout_fl_enabled() is True

    def test_rollout_fl_disabled(self):
        env = {k: v for k, v in os.environ.items() if k != "VLLM_FL_PREFER"}
        with patch.dict(os.environ, env, clear=True):
            assert FLEnvManager.is_rollout_fl_enabled() is False


class TestFLEnvManagerFlagGems:
    """Tests for FlagGems enabled check."""

    @pytest.mark.parametrize("value", ["true", "1", "yes", "True", "YES"])
    def test_flaggems_enabled(self, value):
        with patch.dict(os.environ, {"USE_FLAGGEMS": value}, clear=False):
            assert FLEnvManager.is_flaggems_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", ""])
    def test_flaggems_disabled(self, value):
        with patch.dict(os.environ, {"USE_FLAGGEMS": value}, clear=False):
            assert FLEnvManager.is_flaggems_enabled() is False

    def test_flaggems_disabled_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "USE_FLAGGEMS"}
        with patch.dict(os.environ, env, clear=True):
            assert FLEnvManager.is_flaggems_enabled() is False


class TestFLEnvManagerFlagCX:
    """Tests for FlagCX enabled check."""

    def test_flagcx_enabled_when_path_set(self):
        with patch.dict(os.environ, {"FLAGCX_PATH": "/opt/flagcx"}, clear=False):
            assert FLEnvManager.is_flagcx_enabled() is True

    def test_flagcx_disabled_when_path_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "FLAGCX_PATH"}
        with patch.dict(os.environ, env, clear=True):
            assert FLEnvManager.is_flagcx_enabled() is False


class TestFLEnvManagerGetEnv:
    """Tests for get_training_env and get_rollout_env."""

    def test_get_training_env_returns_set_vars(self):
        test_env = {"TE_FL_PREFER": "flagos", "TE_FL_STRICT": "1", "USE_FLAGGEMS": "true"}
        with patch.dict(os.environ, test_env, clear=False):
            env = FLEnvManager.get_training_env()
            assert env["TE_FL_PREFER"] == "flagos"
            assert env["TE_FL_STRICT"] == "1"
            assert env["USE_FLAGGEMS"] == "true"

    def test_get_training_env_excludes_unset_vars(self):
        env_keys_to_clear = FLEnvManager.TRAINING_ENV_KEYS + FLEnvManager.COMMON_ENV_KEYS
        env = {k: v for k, v in os.environ.items() if k not in env_keys_to_clear}
        with patch.dict(os.environ, env, clear=True):
            result = FLEnvManager.get_training_env()
            assert result == {}

    def test_get_rollout_env_returns_set_vars(self):
        test_env = {"VLLM_FL_PREFER": "flagos", "VLLM_FL_PLATFORM": "cuda"}
        with patch.dict(os.environ, test_env, clear=False):
            env = FLEnvManager.get_rollout_env()
            assert env["VLLM_FL_PREFER"] == "flagos"
            assert env["VLLM_FL_PLATFORM"] == "cuda"

    def test_get_rollout_env_includes_common_keys(self):
        test_env = {"USE_FLAGGEMS": "true", "VLLM_FL_PREFER": "flagos"}
        with patch.dict(os.environ, test_env, clear=False):
            env = FLEnvManager.get_rollout_env()
            assert "USE_FLAGGEMS" in env

    def test_training_and_rollout_env_do_not_overlap(self):
        """Training-specific keys should not appear in rollout env and vice versa."""
        training_specific = set(FLEnvManager.TRAINING_ENV_KEYS)
        rollout_specific = set(FLEnvManager.ROLLOUT_ENV_KEYS)
        assert training_specific.isdisjoint(rollout_specific), "Training and rollout keys should not overlap"


class TestFLEnvManagerWhiteBlacklist:
    """Tests for get_flaggems_whitelist and get_flaggems_blacklist."""

    def test_training_whitelist(self):
        with patch.dict(os.environ, {"TRAINING_FL_FLAGOS_WHITELIST": "rmsnorm,layernorm,softmax"}, clear=False):
            result = FLEnvManager.get_flaggems_whitelist(phase="training")
            assert result == ["rmsnorm", "layernorm", "softmax"]

    def test_rollout_whitelist(self):
        with patch.dict(os.environ, {"VLLM_FL_FLAGOS_WHITELIST": "rmsnorm,silu_and_mul"}, clear=False):
            result = FLEnvManager.get_flaggems_whitelist(phase="rollout")
            assert result == ["rmsnorm", "silu_and_mul"]

    def test_whitelist_strips_spaces(self):
        with patch.dict(os.environ, {"TRAINING_FL_FLAGOS_WHITELIST": " rmsnorm , layernorm "}, clear=False):
            result = FLEnvManager.get_flaggems_whitelist(phase="training")
            assert result == ["rmsnorm", "layernorm"]

    def test_whitelist_returns_none_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "TRAINING_FL_FLAGOS_WHITELIST"}
        with patch.dict(os.environ, env, clear=True):
            assert FLEnvManager.get_flaggems_whitelist(phase="training") is None

    def test_whitelist_returns_none_for_empty_string(self):
        with patch.dict(os.environ, {"TRAINING_FL_FLAGOS_WHITELIST": ""}, clear=False):
            assert FLEnvManager.get_flaggems_whitelist(phase="training") is None

    def test_training_blacklist(self):
        with patch.dict(os.environ, {"TRAINING_FL_FLAGOS_BLACKLIST": "to_copy,zeros"}, clear=False):
            result = FLEnvManager.get_flaggems_blacklist(phase="training")
            assert result == ["to_copy", "zeros"]

    def test_rollout_blacklist(self):
        with patch.dict(os.environ, {"VLLM_FL_FLAGOS_BLACKLIST": "mm,bmm"}, clear=False):
            result = FLEnvManager.get_flaggems_blacklist(phase="rollout")
            assert result == ["mm", "bmm"]

    def test_blacklist_returns_none_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "TRAINING_FL_FLAGOS_BLACKLIST"}
        with patch.dict(os.environ, env, clear=True):
            assert FLEnvManager.get_flaggems_blacklist(phase="training") is None


class TestFLEnvManagerSummary:
    """Tests for get_summary."""

    def test_summary_disabled(self):
        env_keys = FLEnvManager.TRAINING_ENV_KEYS + FLEnvManager.ROLLOUT_ENV_KEYS + FLEnvManager.COMMON_ENV_KEYS
        env = {k: v for k, v in os.environ.items() if k not in env_keys}
        with patch.dict(os.environ, env, clear=True):
            assert FLEnvManager.get_summary() == "FL[disabled]"

    def test_summary_with_training(self):
        env_keys = FLEnvManager.TRAINING_ENV_KEYS + FLEnvManager.ROLLOUT_ENV_KEYS + FLEnvManager.COMMON_ENV_KEYS
        env = {k: v for k, v in os.environ.items() if k not in env_keys}
        env["TE_FL_PREFER"] = "flagos"
        with patch.dict(os.environ, env, clear=True):
            summary = FLEnvManager.get_summary()
            assert "training" in summary
            assert "FL[" in summary

    def test_summary_with_rollout(self):
        env_keys = FLEnvManager.TRAINING_ENV_KEYS + FLEnvManager.ROLLOUT_ENV_KEYS + FLEnvManager.COMMON_ENV_KEYS
        env = {k: v for k, v in os.environ.items() if k not in env_keys}
        env["VLLM_FL_PREFER"] = "flagos"
        with patch.dict(os.environ, env, clear=True):
            summary = FLEnvManager.get_summary()
            assert "rollout" in summary

    def test_summary_with_flaggems(self):
        env_keys = FLEnvManager.TRAINING_ENV_KEYS + FLEnvManager.ROLLOUT_ENV_KEYS + FLEnvManager.COMMON_ENV_KEYS
        env = {k: v for k, v in os.environ.items() if k not in env_keys}
        env["USE_FLAGGEMS"] = "true"
        with patch.dict(os.environ, env, clear=True):
            summary = FLEnvManager.get_summary()
            assert "FlagGems" in summary

    def test_summary_with_flagcx(self):
        env_keys = FLEnvManager.TRAINING_ENV_KEYS + FLEnvManager.ROLLOUT_ENV_KEYS + FLEnvManager.COMMON_ENV_KEYS
        env = {k: v for k, v in os.environ.items() if k not in env_keys}
        env["FLAGCX_PATH"] = "/opt/flagcx"
        with patch.dict(os.environ, env, clear=True):
            summary = FLEnvManager.get_summary()
            assert "FlagCX" in summary

    def test_summary_with_all_enabled(self):
        env_keys = FLEnvManager.TRAINING_ENV_KEYS + FLEnvManager.ROLLOUT_ENV_KEYS + FLEnvManager.COMMON_ENV_KEYS
        env = {k: v for k, v in os.environ.items() if k not in env_keys}
        env.update(
            {
                "TE_FL_PREFER": "flagos",
                "VLLM_FL_PREFER": "flagos",
                "USE_FLAGGEMS": "true",
                "FLAGCX_PATH": "/opt/flagcx",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            summary = FLEnvManager.get_summary()
            assert "training" in summary
            assert "rollout" in summary
            assert "FlagGems" in summary
            assert "FlagCX" in summary


class TestFLEnvManagerPhase:
    """Tests for get_current_phase."""

    def test_current_phase_default_none(self):
        FLEnvManager._current_phase = None
        assert FLEnvManager.get_current_phase() is None

    def test_current_phase_training(self):
        FLEnvManager._current_phase = "training"
        assert FLEnvManager.get_current_phase() == "training"
        FLEnvManager._current_phase = None

    def test_current_phase_rollout(self):
        FLEnvManager._current_phase = "rollout"
        assert FLEnvManager.get_current_phase() == "rollout"
        FLEnvManager._current_phase = None


class TestMayEnableFlagGems:
    """Tests for may_enable_flag_gems function."""

    def test_may_enable_flag_gems_returns_early(self):
        """Current implementation returns early (TODO in code), should not raise."""
        may_enable_flag_gems(phase="training")

    def test_may_enable_flag_gems_rollout_returns_early(self):
        may_enable_flag_gems(phase="rollout")


class TestFLEnvManagerEnvKeys:
    """Tests for environment key definitions."""

    def test_training_env_keys_are_strings(self):
        for key in FLEnvManager.TRAINING_ENV_KEYS:
            assert isinstance(key, str)

    def test_rollout_env_keys_are_strings(self):
        for key in FLEnvManager.ROLLOUT_ENV_KEYS:
            assert isinstance(key, str)

    def test_common_env_keys_are_strings(self):
        for key in FLEnvManager.COMMON_ENV_KEYS:
            assert isinstance(key, str)

    def test_vllm_fl_config_in_rollout_keys(self):
        assert "VLLM_FL_CONFIG" in FLEnvManager.ROLLOUT_ENV_KEYS

    def test_te_fl_prefer_in_training_keys(self):
        assert "TE_FL_PREFER" in FLEnvManager.TRAINING_ENV_KEYS

    def test_use_flaggems_in_common_keys(self):
        assert "USE_FLAGGEMS" in FLEnvManager.COMMON_ENV_KEYS

    def test_use_flagcx_in_common_keys(self):
        assert "USE_FLAGCX" in FLEnvManager.COMMON_ENV_KEYS
