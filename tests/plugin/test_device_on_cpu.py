# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for verl.utils.device module."""

import os
from unittest.mock import MagicMock, patch


class TestGetNcclBackend:
    """Tests for get_nccl_backend with FlagCX support."""

    def test_flagcx_backend_when_enabled(self):
        with patch.dict(os.environ, {"USE_FLAGCX": "1"}, clear=False):
            from verl.utils.device import get_nccl_backend

            assert get_nccl_backend() == "flagcx"

    def test_nccl_backend_default(self):
        env = {k: v for k, v in os.environ.items() if k != "USE_FLAGCX"}
        with patch.dict(os.environ, env, clear=True):
            from verl.utils.device import get_nccl_backend, is_npu_available

            result = get_nccl_backend()
            if is_npu_available:
                assert result == "hccl"
            else:
                assert result == "nccl"

    def test_flagcx_not_enabled_when_zero(self):
        with patch.dict(os.environ, {"USE_FLAGCX": "0"}, clear=False):
            from verl.utils.device import get_nccl_backend

            result = get_nccl_backend()
            assert result != "flagcx"


class TestGetDeviceName:
    """Tests for get_device_name."""

    def test_returns_string(self):
        from verl.utils.device import get_device_name

        result = get_device_name()
        assert isinstance(result, str)
        assert result in ("cuda", "npu", "cpu")


class TestGetVisibleDevicesKeyword:
    """Tests for get_visible_devices_keyword."""

    def test_returns_valid_keyword(self):
        from verl.utils.device import get_visible_devices_keyword

        result = get_visible_devices_keyword()
        assert result in ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")


class TestAutoSetDevice:
    """Tests for auto_set_device."""

    def test_auto_set_device_with_none_config(self):
        from verl.utils.device import auto_set_device

        # Should not raise
        auto_set_device(None)

    def test_auto_set_device_with_mock_config(self):
        from verl.utils.device import auto_set_device

        config = MagicMock()
        config.trainer.device = "cuda"
        # Should not raise
        auto_set_device(config)
