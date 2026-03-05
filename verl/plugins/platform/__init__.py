# Copyright (c) 2026 BAAI. All rights reserved.

"""verl multi-chip platform plugin system.

Provides a ``PlatformRegistry`` singleton that auto-detects the current
hardware platform (or reads the ``VERL_DEVICE_BACKEND`` environment variable)
and exposes a unified device API.

Quick start::

    from verl.plugins.platform import get_platform
    platform = get_platform()
    print(platform.info.device_name)   # "cuda", "npu", ...

Backward-compatible convenience functions (matching the old
``verl.utils.device`` API) are also exported here so that
``verl/utils/device.py`` can be a thin re-export wrapper.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .base import BasePlatform, PlatformInfo

logger = logging.getLogger(__name__)

__all__ = [
    # Core
    "PlatformInfo",
    "BasePlatform",
    "PlatformRegistry",
    "get_platform",
    # Convenience (backward-compat with verl.utils.device)
    "get_device_name",
    "get_torch_device",
    "get_device_id",
    "get_nccl_backend",
    "get_visible_devices_keyword",
    "is_cuda_available",
    "is_npu_available",
    "set_expandable_segments",
    "get_device_capability",
    "auto_set_device",
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PlatformRegistry:
    """Singleton registry for hardware platform backends.

    Platforms register themselves via the ``@PlatformRegistry.register``
    decorator.  At runtime, ``get_platform()`` returns the single active
    platform instance, chosen either by the ``VERL_DEVICE_BACKEND``
    environment variable or by auto-detection priority order.
    """

    _platforms: dict[str, type[BasePlatform]] = {}
    _current: Optional[BasePlatform] = None

    # Auto-detection priority (highest first).
    _detect_order: list[str] = ["cuda", "npu", "mlu", "musa"]

    @classmethod
    def register(cls, device_name: str):
        """Class decorator that registers a platform implementation.

        Usage::

            @PlatformRegistry.register("cuda")
            class CUDAPlatform(BasePlatform):
                ...
        """

        def decorator(platform_cls: type[BasePlatform]) -> type[BasePlatform]:
            cls._platforms[device_name] = platform_cls
            return platform_cls

        return decorator

    @classmethod
    def get_platform(cls) -> BasePlatform:
        """Return the active platform (singleton).

        Resolution order:
        1. ``VERL_DEVICE_BACKEND`` environment variable (e.g. ``"npu"``).
        2. Auto-detect by probing each registered platform in priority order.
        3. Fall back to a ``"cpu"`` stub if nothing is found.
        """
        if cls._current is not None:
            return cls._current

        backend = os.environ.get("VERL_DEVICE_BACKEND")
        if backend:
            backend = backend.lower()
            if backend not in cls._platforms:
                raise ValueError(
                    f"VERL_DEVICE_BACKEND={backend!r} is not a registered platform. "
                    f"Available: {list(cls._platforms.keys())}"
                )
            cls._current = cls._platforms[backend]()
            logger.info("Platform selected via VERL_DEVICE_BACKEND: %s", backend)
            return cls._current

        # Auto-detect
        for name in cls._detect_order:
            if name in cls._platforms:
                instance = cls._platforms[name]()
                if instance.is_available():
                    cls._current = instance
                    logger.info("Platform auto-detected: %s", name)
                    return cls._current

        # Fallback: CPU-only stub
        cls._current = _CPUPlatform()
        logger.info("No accelerator detected, falling back to CPU")
        return cls._current

    @classmethod
    def reset(cls) -> None:
        """Reset the cached platform (mainly for testing)."""
        cls._current = None


# ---------------------------------------------------------------------------
# CPU fallback (not registered — used only as last resort)
# ---------------------------------------------------------------------------


class _CPUPlatform(BasePlatform):
    info = PlatformInfo(
        vendor_name="generic",
        device_name="cpu",
        device_query_cmd="",
        comm_backend="gloo",
        visible_devices_env="",
    )

    def is_available(self) -> bool:
        return True

    def get_torch_device_module(self):
        # torch.cpu exists since PyTorch 2.0
        return getattr(__import__("torch"), "cpu", None)

    def current_device_id(self) -> int:
        return 0

    def device_count(self) -> int:
        return 1

    def set_device(self, device_id: int) -> None:
        pass

    def manual_seed(self, seed: int) -> None:
        __import__("torch").manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        pass

    def empty_cache(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def memory_allocated(self, device=None) -> int:
        return 0

    def max_memory_allocated(self, device=None) -> int:
        return 0


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def get_platform() -> BasePlatform:
    """Shortcut for ``PlatformRegistry.get_platform()``."""
    return PlatformRegistry.get_platform()


# ---------------------------------------------------------------------------
# Backward-compatible convenience functions (matching verl.utils.device API)
# ---------------------------------------------------------------------------


def get_device_name() -> str:
    return get_platform().info.device_name


def get_torch_device():
    return get_platform().get_torch_device_module()


def get_device_id() -> int:
    return get_platform().current_device_id()


def get_nccl_backend() -> str:
    return get_platform().info.comm_backend


def get_visible_devices_keyword() -> str:
    return get_platform().info.visible_devices_env


def is_cuda_available() -> bool:
    from .cuda import CUDAPlatform

    return CUDAPlatform().is_available()


def is_npu_available() -> bool:
    from .npu import NPUPlatform

    return NPUPlatform().is_available()


def set_expandable_segments(enable: bool) -> None:
    get_platform().set_expandable_segments(enable)


def get_device_capability(device_id: int = 0) -> tuple[Optional[int], Optional[int]]:
    return get_platform().get_device_capability(device_id)


def auto_set_device(config) -> None:
    """Automatically set ``config.trainer.device`` to match the detected platform."""
    if config and hasattr(config, "trainer") and hasattr(config.trainer, "device"):
        detected = get_device_name()
        if detected != "cpu" and config.trainer.device != detected:
            if config.trainer.device != "cpu":
                logger.warning(
                    "Detected %s device but config.trainer.device=%s, automatically setting to %r.",
                    detected,
                    config.trainer.device,
                    detected,
                )
            config.trainer.device = detected


# ---------------------------------------------------------------------------
# Auto-register built-in platforms on import
# ---------------------------------------------------------------------------


def _register_builtin_platforms() -> None:
    """Import built-in platform modules so they self-register."""
    from . import cuda, npu  # noqa: F401

    # MLU / MUSA are optional — only register if the driver package exists
    try:
        from . import mlu  # noqa: F401
    except Exception:
        pass
    try:
        from . import musa  # noqa: F401
    except Exception:
        pass


# Perform registration with decorators applied during import
from .cuda import CUDAPlatform  # noqa: E402
from .npu import NPUPlatform  # noqa: E402

PlatformRegistry.register("cuda")(CUDAPlatform)
PlatformRegistry.register("npu")(NPUPlatform)

try:
    from .mlu import MLUPlatform  # noqa: E402

    PlatformRegistry.register("mlu")(MLUPlatform)
except Exception:
    pass

try:
    from .musa import MUSAPlatform  # noqa: E402

    PlatformRegistry.register("musa")(MUSAPlatform)
except Exception:
    pass
