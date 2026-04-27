# Copyright (c) 2026 BAAI. All rights reserved.
# Adopted from https://github.com/microsoft/DeepSpeed/blob/master/accelerator/abstract_accelerator.py
"""Abstract base class defining the platform interface for device backends.

To add support for a new chip/accelerator, subclass ``PlatformBase`` and
implement all abstract methods.  Then register the platform name in
``platform_manager.py`` so that auto-detection or ``VERL_PLATFORM`` can
pick it up.
"""

import abc
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional


class PlatformBase(abc.ABC):
    """Hardware-agnostic interface for accelerator backends.

    Every concrete platform (CUDA, NPU, CPU, XPU, …) must implement the
    methods below so that the rest of the verl codebase can remain
    device-agnostic.
    """

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def device_name(self) -> str:
        """Return the device type string (e.g. ``'cuda'``, ``'npu'``, ``'cpu'``)."""
        ...

    @property
    @abc.abstractmethod
    def device_module(self) -> ModuleType:
        """Return the ``torch.<device>`` namespace module (e.g. ``torch.cuda``)."""
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the accelerator is available on this host."""
        ...

    @abc.abstractmethod
    def current_device(self) -> int:
        """Return the index of the currently selected device."""
        ...

    @abc.abstractmethod
    def device_count(self) -> int:
        """Return the number of available devices of this type."""
        ...

    @abc.abstractmethod
    def set_device(self, device_index: int) -> None:
        """Select the device at *device_index*."""
        ...

    @abc.abstractmethod
    def synchronize(self, device_index: Optional[int] = None) -> None:
        """Block until all pending work on the device completes."""
        ...

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def manual_seed(self, seed: int) -> None:
        """Seed the current device's RNG."""
        ...

    @abc.abstractmethod
    def manual_seed_all(self, seed: int) -> None:
        """Seed **all** devices' RNG."""
        ...

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def set_allocator_settings(self, settings: str) -> None:
        """Configure the memory allocator (e.g. expandable segments)."""
        ...

    @abc.abstractmethod
    def empty_cache(self) -> None:
        """Release all unused cached memory."""
        ...

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        """Return ``(major, minor)`` compute capability, or ``(None, None)``."""
        ...

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def communication_backend_name(self) -> str:
        """Return the default collective-communication backend name (e.g. ``'nccl'``)."""
        ...

    @abc.abstractmethod
    def visible_devices_envvar(self) -> str:
        """Return the environment-variable name that controls visible devices."""
        ...

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @abc.abstractmethod
    @contextmanager
    def nvtx_range(self, msg: str):
        """Context manager that wraps a block with an NVTX / profiler range."""
        ...

    @abc.abstractmethod
    def profiler_start(self) -> None:
        """Start the device profiler (no-op on unsupported platforms)."""
        ...

    @abc.abstractmethod
    def profiler_stop(self) -> None:
        """Stop the device profiler (no-op on unsupported platforms)."""
        ...

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def cudart(self) -> Any:
        """Return the CUDA runtime API object, or ``None`` if not applicable."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> None:  # noqa: B027
        """Ensure the hardware backend library is fully loaded.

        Called once by :func:`get_platform` right after the platform singleton
        is created.  Subclasses that depend on a third-party extension module
        (e.g. ``torch_musa``, ``torch_npu``) should override this to import
        the module eagerly so that downstream libraries like ``transformers``
        and ``accelerate`` see a consistent runtime environment.

        The default implementation is a no-op.
        """
        pass
