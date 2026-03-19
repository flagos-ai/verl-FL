# Copyright (c) 2026 BAAI. All rights reserved.
"""Unit tests for the platform abstraction layer.

We pre-populate ``sys.modules`` with lightweight stubs for the top-level
``verl`` and ``verl.plugin`` packages so that importing the platform
sub-package does **not** trigger the heavy dependency chain pulled in by
``verl/__init__.py`` (ray, tensordict, …).
"""

import os
import sys
from types import ModuleType

# ---------------------------------------------------------------------------
# Bootstrap: prevent verl.__init__ from executing (it imports ray, etc.)
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _pkg, _rel in [("verl", "verl"), ("verl.plugin", os.path.join("verl", "plugin"))]:
    if _pkg not in sys.modules:
        _stub = ModuleType(_pkg)
        _stub.__path__ = [os.path.join(_project_root, _rel)]
        _stub.__package__ = _pkg
        sys.modules[_pkg] = _stub

# ---------------------------------------------------------------------------

from unittest import mock  # noqa: E402

import pytest  # noqa: E402

from verl.plugin.platform import get_platform, set_platform  # noqa: E402
from verl.plugin.platform.platform_manager import (  # noqa: E402
    _create_platform,
    _detect_platform_name,
)


class TestPlatformDetection:
    """Test platform auto-detection logic."""

    def setup_method(self):
        """Reset platform singleton before each test."""
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_verl_platform_env_override_cuda(self):
        """Test VERL_PLATFORM environment variable override for CUDA."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "cuda"}):
            name = _detect_platform_name()
            assert name == "cuda", f"Expected 'cuda', got '{name}'"

    def test_verl_platform_env_override_npu(self):
        """Test VERL_PLATFORM environment variable override for NPU."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "npu"}):
            name = _detect_platform_name()
            assert name == "npu", f"Expected 'npu', got '{name}'"

    def test_verl_platform_env_override_cpu(self):
        """Test VERL_PLATFORM environment variable override for CPU."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "cpu"}):
            name = _detect_platform_name()
            assert name == "cpu", f"Expected 'cpu', got '{name}'"

    def test_verl_platform_invalid_value(self):
        """Test that invalid VERL_PLATFORM values fall back to auto-detection."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "invalid"}):
            name = _detect_platform_name()
            assert name in ("cuda", "npu", "cpu"), f"Got invalid platform name: {name}"

    def test_verl_platform_case_insensitive(self):
        """Test that VERL_PLATFORM is case-insensitive."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "CUDA"}):
            name = _detect_platform_name()
            assert name == "cuda", f"Expected 'cuda', got '{name}'"


class TestPlatformCreation:
    """Test platform creation and initialization."""

    def setup_method(self):
        """Reset platform singleton before each test."""
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_create_cpu_platform(self):
        """Test creating CPU platform."""
        platform = _create_platform("cpu")
        assert platform.device_name == "cpu"
        assert platform.is_available() is True

    def test_create_cuda_platform_fallback(self):
        """Test that CUDA platform falls back to CPU if not available."""
        with mock.patch("torch.cuda.is_available", return_value=False):
            platform = _create_platform("cuda")
            assert platform.device_name == "cpu"

    def test_create_invalid_platform(self):
        """Test that invalid platform name raises ValueError."""
        with pytest.raises(ValueError):
            _create_platform("invalid_platform")


class TestPlatformInterface:
    """Test platform interface methods."""

    def setup_method(self):
        """Reset platform singleton before each test."""
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_get_platform_returns_singleton(self):
        """Test that get_platform() returns a singleton."""
        platform1 = get_platform()
        platform2 = get_platform()
        assert platform1 is platform2, "get_platform() should return the same instance"

    def test_set_platform_overrides(self):
        """Test that set_platform() overrides the singleton."""
        from verl.plugin.platform.platform_cpu import PlatformCPU

        custom_platform = PlatformCPU()
        set_platform(custom_platform)

        retrieved = get_platform()
        assert retrieved is custom_platform, "set_platform() should override the singleton"

    def test_manual_seed_cpu(self):
        """Test manual_seed on CPU platform."""
        from verl.plugin.platform.platform_cpu import PlatformCPU

        platform = PlatformCPU()

        platform.manual_seed(42)
        platform.manual_seed_all(42)

    def test_device_capability_cpu(self):
        """Test get_device_capability on CPU platform returns (None, None)."""
        from verl.plugin.platform.platform_cpu import PlatformCPU

        platform = PlatformCPU()

        major, minor = platform.get_device_capability()
        assert major is None and minor is None

    def test_communication_backend_cpu(self):
        """Test communication_backend_name on CPU platform returns 'gloo'."""
        from verl.plugin.platform.platform_cpu import PlatformCPU

        platform = PlatformCPU()

        backend = platform.communication_backend_name()
        assert backend == "gloo"

    def test_nvtx_range_context_manager(self):
        """Test that nvtx_range works as context manager."""
        from verl.plugin.platform.platform_cpu import PlatformCPU

        platform = PlatformCPU()

        with platform.nvtx_range("test_range"):
            pass


class TestCPUPlatformDeviceCount:
    """Test PlatformCPU.device_count() bug fix."""

    def test_device_count_with_local_size(self):
        """Test device_count when LOCAL_SIZE is set."""
        from verl.plugin.platform.platform_cpu import PlatformCPU

        with mock.patch.dict(os.environ, {"LOCAL_SIZE": "4"}):
            platform = PlatformCPU()
            count = platform.device_count()
            assert count == 4, f"Expected 4, got {count}"

    def test_device_count_without_local_size(self):
        """Test device_count falls back to os.cpu_count() when LOCAL_SIZE is not set."""
        from verl.plugin.platform.platform_cpu import PlatformCPU

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("os.cpu_count", return_value=8):
                platform = PlatformCPU()
                count = platform.device_count()
                assert count == 8, f"Expected 8, got {count}"

    def test_device_count_fallback_to_one(self):
        """Test device_count falls back to 1 when os.cpu_count() returns None."""
        from verl.plugin.platform.platform_cpu import PlatformCPU

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("os.cpu_count", return_value=None):
                platform = PlatformCPU()
                count = platform.device_count()
                assert count >= 1, f"Expected at least 1, got {count}"

    def test_device_count_no_default_zero(self):
        """Test that device_count never returns 0."""
        from verl.plugin.platform.platform_cpu import PlatformCPU

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("os.cpu_count", return_value=None):
                platform = PlatformCPU()
                count = platform.device_count()
                assert count != 0, "device_count should never return 0"


class TestEnvironmentVariableValidation:
    """Test VERL_PLATFORM environment variable validation."""

    def setup_method(self):
        """Reset platform singleton before each test."""
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_invalid_platform_falls_back(self):
        """Test that invalid VERL_PLATFORM values fall back to auto-detection."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "opencl"}):
            name = _detect_platform_name()
            # Should fall back, not return "opencl"
            assert name != "opencl"
            assert name in ("cuda", "npu", "cpu")

    def test_empty_platform_triggers_auto_detection(self):
        """Test that empty VERL_PLATFORM triggers auto-detection."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": ""}):
            name = _detect_platform_name()
            assert name in ("cuda", "npu", "cpu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
