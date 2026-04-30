# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vllm-plugin-FL/vllm_fl/distributed/device_communicators/flagcx.py

"""
Standalone FlagCX communicator for cross-device weight synchronization.

Uses the official FlagCX Python wrapper (from FLAGCX_PATH) directly,
bypassing torch.distributed entirely.
"""

import ctypes
import logging
import os
import sys
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Import the official FlagCX wrapper from FLAGCX_PATH
_flagcx_path = os.getenv("FLAGCX_PATH")
if _flagcx_path and os.path.isdir(_flagcx_path):
    if _flagcx_path not in sys.path:
        sys.path.append(_flagcx_path)

from plugin.interservice.flagcx_wrapper import (  # noqa: E402
    FLAGCXLibrary,
    buffer_type,
    flagcxDataTypeEnum,
    flagcxStream_t,
    flagcxUniqueId,
)


def _current_stream(device: torch.device):
    """Get the current stream for the given device type."""
    if device.type == "musa":
        return torch.musa.current_stream(device)
    elif device.type == "cuda":
        return torch.cuda.current_stream(device)
    else:
        raise RuntimeError(f"Unsupported device type: {device.type}")


def _detect_local_device_type() -> str:
    """Detect actual accelerator on this node via tensor allocation probe."""
    try:
        if hasattr(torch, "musa") and callable(getattr(torch.musa, "is_available", None)):
            if torch.musa.is_available():
                t = torch.zeros(1, device="musa:0")
                del t
                return "musa"
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            t = torch.zeros(1, device="cuda:0")
            del t
            return "cuda"
    except Exception:
        pass
    raise RuntimeError("No CUDA or MUSA device found on this node.")


class PyFlagcxCommunicator:
    """
    FlagCX communicator that bypasses torch.distributed.

    Uses StatelessProcessGroup (TCP) for rendezvous/unique_id distribution,
    then communicates via FlagCX C API directly.
    """

    def __init__(
        self,
        group,  # StatelessProcessGroup
        device: str | int | torch.device,
        library_path: Optional[str] = None,
    ):
        self.rank = group.rank
        self.world_size = group.world_size
        self.group = group

        if self.world_size == 1:
            self.disabled = True
            return
        self.disabled = False

        # Load FlagCX library
        if library_path is None:
            flagcx_path = os.getenv("FLAGCX_PATH", "")
            library_path = os.path.join(flagcx_path, "build/lib/libflagcx.so")
        self.flagcx = FLAGCXLibrary(library_path)

        # Generate and distribute unique_id
        if self.rank == 0:
            self.unique_id = self.flagcx.flagcxGetUniqueId().contents
        else:
            self.unique_id = flagcxUniqueId()
        self.unique_id = group.broadcast_obj(self.unique_id, src=0)

        # Resolve device
        if isinstance(device, int):
            device_type = _detect_local_device_type()
            device = torch.device(f"{device_type}:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Initialize communicator under the correct device context
        if self.device.type == "musa":
            device_ctx = torch.musa.device(self.device)
        else:
            device_ctx = torch.cuda.device(self.device)

        with device_ctx:
            self.comm = self.flagcx.flagcxCommInitRank(self.world_size, ctypes.byref(self.unique_id), self.rank)
            # flagcxCommInitRank internally calls setDevice which may corrupt
            # the CUDA/MUSA runtime state. Re-set explicitly.
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            elif self.device.type == "musa":
                torch.musa.set_device(self.device)
            # Warmup
            data = torch.zeros(1, device=self.device)
            self.broadcast(data, src=0)
            _current_stream(self.device).synchronize()
            del data

        logger.info(
            f"PyFlagcxCommunicator rank={self.rank}, world_size={self.world_size}, device={self.device} initialized"
        )

    def _stream_copy(self, torch_stream):
        """Wrap adaptor_stream_copy to handle both cuda_stream and musa_stream."""
        new_stream = flagcxStream_t()
        if hasattr(torch_stream, "cuda_stream"):
            raw = torch_stream.cuda_stream
        elif hasattr(torch_stream, "musa_stream"):
            raw = torch_stream.musa_stream
        else:
            raise AttributeError(f"Cannot get raw stream pointer from {type(torch_stream)}")
        self.flagcx.handler.contents.devHandle.contents.streamCopy(ctypes.byref(new_stream), ctypes.c_void_p(raw))
        return new_stream

    def broadcast(self, tensor: torch.Tensor, src: int = 0, stream=None) -> None:
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"FlagCX communicator bound to {self.device}, but tensor is on {tensor.device}"
        )
        if stream is None:
            stream = _current_stream(self.device)
        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())
        flagcx_stream = self._stream_copy(stream)
        self.flagcx.flagcxBroadcast(
            sendbuff,
            recvbuff,
            tensor.numel(),
            flagcxDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            flagcx_stream,
        )
        self.flagcx.adaptor_stream_free(flagcx_stream)
