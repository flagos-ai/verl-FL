# Copyright (c) 2026 BAAI. All rights reserved.
"""Moore Threads MUSA platform implementation."""

import logging
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase

logger = logging.getLogger(__name__)


def _get_musa_module() -> ModuleType:
    """Return the ``torch.musa`` module, importing ``torch_musa`` if needed."""
    if not hasattr(torch, "musa"):
        try:
            import torch_musa  # noqa: F401 – registers torch.musa
        except (ImportError, RuntimeError, AttributeError) as err:
            raise ImportError(
                "Moore Threads MUSA platform requires the 'torch_musa' package. Please install it first."
            ) from err
    return torch.musa


class PlatformMUSA(PlatformBase):
    """Platform backend for Moore Threads MUSA GPUs."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "musa"

    @property
    def device_module(self) -> ModuleType:
        return _get_musa_module()

    def is_available(self) -> bool:
        try:
            musa = _get_musa_module()
            return musa.is_available()
        except ImportError:
            return False

    def current_device(self) -> int:
        return _get_musa_module().current_device()

    def device_count(self) -> int:
        return _get_musa_module().device_count()

    def set_device(self, device_index: int) -> None:
        _get_musa_module().set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if device_index is not None:
            _get_musa_module().synchronize(device_index)
        else:
            _get_musa_module().synchronize()

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        _get_musa_module().manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        _get_musa_module().manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        musa = _get_musa_module()
        if hasattr(musa, "memory") and hasattr(musa.memory, "_set_allocator_settings"):
            musa.memory._set_allocator_settings(settings)

    def empty_cache(self) -> None:
        _get_musa_module().empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        musa = _get_musa_module()
        if hasattr(musa, "get_device_capability"):
            return musa.get_device_capability(device_index)
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        if os.getenv("USE_FLAGCX", "").lower() in ("1", "true"):
            return "flagcx"
        return "mccl"

    def visible_devices_envvar(self) -> str:
        return "MUSA_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        logger.debug("NVTX range (no-op on MUSA): %s", msg)
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_flagcx_stream() -> None:
        """Expose a ``cuda_stream`` property on ``torch.musa.Stream``.

        FlagCX's ``adaptor_stream_copy`` hardcodes ``stream.cuda_stream``,
        but MUSA streams use ``musa_stream``.  This alias keeps FlagCX
        working without modifying its upstream sources.
        """
        musa = _get_musa_module()
        stream_cls = getattr(musa, "Stream", None)
        if stream_cls is not None and not hasattr(stream_cls, "cuda_stream"):
            stream_cls.cuda_stream = property(lambda self: self.musa_stream)
            logger.debug("FlagCX stream.cuda_stream alias applied")

    @staticmethod
    def _patch_module_cuda() -> None:
        """Redirect ``torch.nn.Module.cuda()`` and ``torch.Tensor.cuda()`` to the
        MUSA device.

        Third-party libraries like ``mbridge`` hardcode
        ``model.cuda(torch.cuda.current_device())``.  We intercept the ``.cuda()``
        method itself and redirect it to ``self.to("musa:...")``.

        ``torch.cuda`` module is **not** patched here, so
        ``torch.cuda.is_available()`` stays truthful on this platform and does not
        trick Megatron-LM-FL's platform auto-detection.
        """
        musa = _get_musa_module()

        def _musa_module_cuda(self, device=None, **kwargs):
            if device is None:
                device = musa.current_device()
            if isinstance(device, int):
                device = torch.device("musa", device)
            return self.to(device, **{k: v for k, v in kwargs.items() if k != "non_blocking"})

        if hasattr(torch.nn.Module, "cuda"):
            torch.nn.Module.cuda = _musa_module_cuda
        if hasattr(torch.Tensor, "cuda"):
            torch.Tensor.cuda = _musa_module_cuda
        logger.debug("Module.cuda / Tensor.cuda redirected to musa")

    @staticmethod
    def _patch_sglang_launch() -> None:
        """Restore the ``_launch_subprocesses`` symbol on the SGLang HTTP server
        module for backward compatibility.

        Newer SGLang versions moved ``_launch_subprocesses`` from
        ``sglang.srt.entrypoints.http_server`` into
        ``Engine._launch_subprocesses`` with an extended signature.  This shim
        wraps the new API so that ``async_sglang_server.py`` (which vendors may
        not control) works without modification.
        """
        try:
            import sglang.srt.entrypoints.http_server as _sglang_http
            from sglang.srt.entrypoints.engine import Engine as _SglangEngine
            from sglang.srt.entrypoints.engine import init_tokenizer_manager as _init_tok_mgr
            from sglang.srt.entrypoints.http_server import (
                run_detokenizer_process as _run_detok,
                run_scheduler_process as _run_sched,
            )
        except ImportError:
            logger.debug("sglang not available; skipping SGLang launch patch")
            return

        if hasattr(_sglang_http, "_launch_subprocesses"):
            logger.debug("SGLang _launch_subprocesses already present; skipping")
            return

        def _patched_launch_subprocesses(server_args):
            """Wrap Engine._launch_subprocesses and return the legacy
            3-element tuple that ``async_sglang_server.py`` expects."""
            (
                tokenizer_manager,
                template_manager,
                _port_args,          # new API only
                sched_result,        # new API only — contains scheduler_infos
                _watchdog,           # new API only
            ) = _SglangEngine._launch_subprocesses(
                server_args=server_args,
                init_tokenizer_manager_func=_init_tok_mgr,
                run_scheduler_process_func=_run_sched,
                run_detokenizer_process_func=_run_detok,
            )
            return (
                tokenizer_manager,
                template_manager,
                sched_result.scheduler_infos,
            )

        _sglang_http._launch_subprocesses = _patched_launch_subprocesses
        logger.debug("SGLang _launch_subprocesses shim applied")

    @staticmethod
    def _patch_sglang_torch() -> None:
        """Patch SGLang's ``patch_torch._device_to_uuid`` /
        ``_device_from_maybe_uuid`` for MUSA.

        Called from verl processes (e.g. WorkerDict weight-sync) where the
        ``sglang_fl`` plugin may not be loaded.  The plugin covers SGLang
        subprocesses (scheduler / tp-worker) via ``os.register_at_fork``;
        this covers the verl side.
        """
        try:
            import sglang.srt.utils.patch_torch as _pt
        except ImportError:
            return

        _orig_to_uuid = _pt._device_to_uuid
        _orig_from_uuid = _pt._device_from_maybe_uuid

        def _patched_to_uuid(device: int) -> str:
            try:
                return _orig_to_uuid(device)
            except AttributeError:
                return str(device)

        def _patched_from_uuid(device_maybe_uuid):
            try:
                return _orig_from_uuid(device_maybe_uuid)
            except AttributeError:
                if isinstance(device_maybe_uuid, int):
                    return device_maybe_uuid
                if isinstance(device_maybe_uuid, str):
                    return int(device_maybe_uuid)
                raise TypeError(f"Unknown type: {type(device_maybe_uuid)}")

        _pt._device_to_uuid = _patched_to_uuid
        _pt._device_from_maybe_uuid = _patched_from_uuid
        logger.debug("MUSA patch_torch device_uuid patched (verl side)")

    @staticmethod
    def _patch_metrics_reduce() -> None:
        """Patch ``verl.utils.metric.utils.reduce_metrics`` to move tensors
        to CPU before numpy operations — identical logic to the vendored
        ``_to_cpu`` helper in that module.

        Because ``reduce_metrics`` is imported by reference in several
        modules (e.g. ``ray_trainer``), we also update the reference in
        ``sys.modules`` where the caller has already been imported.
        """
        import sys

        import torch

        # -- copied from verl.utils.metric.utils._to_cpu ------------------
        def _to_cpu(val):
            if torch.is_tensor(val):
                return val.detach().cpu()
            if isinstance(val, (list, tuple)):
                return type(val)(_to_cpu(v) for v in val)
            if isinstance(val, dict):
                return {k: _to_cpu(v) for k, v in val.items()}
            return val

        try:
            from verl.utils.metric import utils as _mu
        except ImportError:
            logger.debug("verl.utils.metric.utils not available; skip metrics patch")
            return

        _orig_reduce = _mu.reduce_metrics

        def _patched_reduce_metrics(metrics):
            # Move tensors to CPU *before* numpy sees them.
            for key in list(metrics.keys()):
                metrics[key] = _to_cpu(metrics[key])
            return _orig_reduce(metrics)

        _mu.reduce_metrics = _patched_reduce_metrics

        # The caller in ray_trainer holds a local "from … import reduce_metrics"
        # reference; update it if the module has already been imported.
        # Also update verl.utils.metric (the package __init__) because
        # "from verl.utils.metric import reduce_metrics" resolves through it.
        for _mod_name in (
            "verl.trainer.ppo.ray_trainer",
            "verl.utils.metric",
            "verl.utils.metric.utils",
        ):
            _mod = sys.modules.get(_mod_name)
            if _mod is not None:
                _mod.reduce_metrics = _patched_reduce_metrics

        logger.debug("MUSA reduce_metrics CPU-move patch applied")

    def ensure_initialized(self) -> None:
        """Eagerly load ``torch_musa`` so that downstream libraries
        (``transformers``, ``accelerate``, ``flash_attn``, …) see a fully
        initialised MUSA runtime when they are imported later.

        Applies vendor-compatibility patches for third-party libraries that
        hardcode CUDA-specific APIs.
        """
        _get_musa_module() # ensure torch_musa is loaded

        PlatformMUSA._patch_flagcx_stream()
        PlatformMUSA._patch_module_cuda()
        PlatformMUSA._patch_sglang_launch()
        PlatformMUSA._patch_sglang_torch()
        PlatformMUSA._patch_metrics_reduce()

        logger.debug("torch_musa initialised by PlatformMUSA.ensure_initialized()")
