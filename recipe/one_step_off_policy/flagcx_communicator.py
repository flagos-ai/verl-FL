# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from:
#   - FlagCX/plugin/interservice/flagcx_wrapper.py (ctypes bindings)
#   - vllm-plugin-FL/vllm_fl/distributed/device_communicators/flagcx.py (communicator)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Standalone FlagCX communicator for cross-device weight synchronization.

This module provides a pure-Python (ctypes) interface to the FlagCX collective
communication library, bypassing torch.distributed entirely.  It is used to
broadcast model weights between actor workers (CUDA) and rollout workers (MUSA)
that live in separate torch.distributed process groups.
"""

import ctypes
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

# =============================================================================
# FlagCX C type definitions
# =============================================================================

flagcxResult_t = ctypes.c_int
flagcxDataType_t = ctypes.c_int
flagcxRedOp_t = ctypes.c_int
flagcxMemcpyType_t = ctypes.c_int
flagcxMemType_t = ctypes.c_int
flagcxEventType_t = ctypes.c_int

flagcxComm_t = ctypes.c_void_p
flagcxEvent_t = ctypes.c_void_p
flagcxStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p


class flagcxUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 256)]


# Device handle function types (only stream ops needed for communicator)
DEVICE_SYNCHRONIZE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t)
DEVICE_MEMCPY_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    flagcxMemcpyType_t, flagcxStream_t
)
DEVICE_MEMSET_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
    flagcxMemType_t, flagcxStream_t
)
DEVICE_MALLOC_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
    flagcxMemType_t, flagcxStream_t
)
DEVICE_FREE_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.c_void_p, flagcxMemType_t, flagcxStream_t
)
SET_DEVICE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.c_int)
GET_DEVICE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(ctypes.c_int))
GET_DEVICE_COUNT_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(ctypes.c_int))
GET_VENDOR_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.c_char_p)
HOST_GET_DEVICE_POINTER_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
)

STREAM_CREATE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(flagcxStream_t))
STREAM_DESTROY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t)
STREAM_COPY_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.POINTER(flagcxStream_t), ctypes.c_void_p
)
STREAM_FREE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t)
STREAM_SYNCHRONIZE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t)
STREAM_QUERY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t)
STREAM_WAIT_EVENT_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t, flagcxEvent_t)

EVENT_CREATE_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.POINTER(flagcxEvent_t), flagcxEventType_t
)
EVENT_DESTROY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxEvent_t)
EVENT_RECORD_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxEvent_t, flagcxStream_t)
EVENT_SYNCHRONIZE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxEvent_t)
EVENT_QUERY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxEvent_t)

IPC_MEM_HANDLE_CREATE_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)
)
IPC_MEM_HANDLE_GET_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.c_void_p, ctypes.c_void_p
)
IPC_MEM_HANDLE_OPEN_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
)
IPC_MEM_HANDLE_CLOSE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.c_void_p)
IPC_MEM_HANDLE_FREE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.c_void_p)


class flagcxDeviceHandle(ctypes.Structure):
    _fields_ = [
        ("deviceSynchronize", DEVICE_SYNCHRONIZE_FUNCTYPE),
        ("deviceMemcpy", DEVICE_MEMCPY_FUNCTYPE),
        ("deviceMemset", DEVICE_MEMSET_FUNCTYPE),
        ("deviceMalloc", DEVICE_MALLOC_FUNCTYPE),
        ("deviceFree", DEVICE_FREE_FUNCTYPE),
        ("setDevice", SET_DEVICE_FUNCTYPE),
        ("getDevice", GET_DEVICE_FUNCTYPE),
        ("getDeviceCount", GET_DEVICE_COUNT_FUNCTYPE),
        ("getVendor", GET_VENDOR_FUNCTYPE),
        ("hostGetDevicePointer", HOST_GET_DEVICE_POINTER_FUNCTYPE),
        ("streamCreate", STREAM_CREATE_FUNCTYPE),
        ("streamDestroy", STREAM_DESTROY_FUNCTYPE),
        ("streamCopy", STREAM_COPY_FUNCTYPE),
        ("streamFree", STREAM_FREE_FUNCTYPE),
        ("streamSynchronize", STREAM_SYNCHRONIZE_FUNCTYPE),
        ("streamQuery", STREAM_QUERY_FUNCTYPE),
        ("streamWaitEvent", STREAM_WAIT_EVENT_FUNCTYPE),
        ("eventCreate", EVENT_CREATE_FUNCTYPE),
        ("eventDestroy", EVENT_DESTROY_FUNCTYPE),
        ("eventRecord", EVENT_RECORD_FUNCTYPE),
        ("eventSynchronize", EVENT_SYNCHRONIZE_FUNCTYPE),
        ("eventQuery", EVENT_QUERY_FUNCTYPE),
        ("ipcMemHandleCreate", IPC_MEM_HANDLE_CREATE_FUNCTYPE),
        ("ipcMemHandleGet", IPC_MEM_HANDLE_GET_FUNCTYPE),
        ("ipcMemHandleOpen", IPC_MEM_HANDLE_OPEN_FUNCTYPE),
        ("ipcMemHandleClose", IPC_MEM_HANDLE_CLOSE_FUNCTYPE),
        ("ipcMemHandleFree", IPC_MEM_HANDLE_FREE_FUNCTYPE),
    ]


flagcxDeviceHandle_t = ctypes.POINTER(flagcxDeviceHandle)


class flagcxHandlerGroup(ctypes.Structure):
    _fields_ = [
        ("uniqueId", ctypes.POINTER(flagcxUniqueId)),
        ("comm", flagcxComm_t),
        ("devHandle", flagcxDeviceHandle_t),
    ]


flagcxHandlerGroup_t = ctypes.POINTER(flagcxHandlerGroup)


# =============================================================================
# Data type mapping
# =============================================================================


class flagcxDataTypeEnum:
    flagcxInt8 = 0
    flagcxUint8 = 1
    flagcxInt32 = 2
    flagcxUint32 = 3
    flagcxInt64 = 4
    flagcxUint64 = 5
    flagcxFloat16 = 6
    flagcxFloat32 = 7
    flagcxFloat64 = 8
    flagcxBfloat16 = 9

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        mapping = {
            torch.int8: cls.flagcxInt8,
            torch.uint8: cls.flagcxUint8,
            torch.int32: cls.flagcxInt32,
            torch.int64: cls.flagcxInt64,
            torch.float16: cls.flagcxFloat16,
            torch.float32: cls.flagcxFloat32,
            torch.float64: cls.flagcxFloat64,
            torch.bfloat16: cls.flagcxBfloat16,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype for FlagCX: {dtype}")
        return mapping[dtype]


# =============================================================================
# FLAGCXLibrary — ctypes wrapper around libflagcx.so
# =============================================================================


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class FLAGCXLibrary:
    """Minimal ctypes wrapper for the FlagCX C library."""

    exported_functions = [
        Function("flagcxHandleInit", flagcxResult_t, [ctypes.POINTER(flagcxHandlerGroup_t)]),
        Function("flagcxHandleFree", flagcxResult_t, [flagcxHandlerGroup_t]),
        Function("flagcxGetErrorString", ctypes.c_char_p, [flagcxResult_t]),
        Function("flagcxGetVersion", flagcxResult_t, [ctypes.POINTER(ctypes.c_int)]),
        Function(
            "flagcxGetUniqueId",
            flagcxResult_t,
            [ctypes.POINTER(ctypes.POINTER(flagcxUniqueId))],
        ),
        Function(
            "flagcxCommInitRank",
            flagcxResult_t,
            [ctypes.POINTER(flagcxComm_t), ctypes.c_int, ctypes.POINTER(flagcxUniqueId), ctypes.c_int],
        ),
        Function(
            "flagcxBroadcast",
            flagcxResult_t,
            [buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t, ctypes.c_int, flagcxComm_t, flagcxStream_t],
        ),
        Function("flagcxCommDestroy", flagcxResult_t, [flagcxComm_t]),
    ]

    # Cache to avoid loading the same .so multiple times
    _lib_cache: Dict[str, Any] = {}
    _func_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            flagcx_path = os.getenv("FLAGCX_PATH", "")
            so_file = os.path.join(flagcx_path, "build", "lib", "libflagcx.so")

        if so_file not in FLAGCXLibrary._lib_cache:
            FLAGCXLibrary._lib_cache[so_file] = ctypes.CDLL(so_file)
        self.lib = FLAGCXLibrary._lib_cache[so_file]

        if so_file not in FLAGCXLibrary._func_cache:
            _funcs: Dict[str, Any] = {}
            for func in FLAGCXLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            FLAGCXLibrary._func_cache[so_file] = _funcs
        self._funcs = FLAGCXLibrary._func_cache[so_file]

        # Initialize the device handler
        self.handler = flagcxHandlerGroup_t()
        self.FLAGCX_CHECK(self._funcs["flagcxHandleInit"](ctypes.byref(self.handler)))

    def __del__(self):
        if hasattr(self, "handler") and self.handler:
            try:
                self.FLAGCX_CHECK(self._funcs["flagcxHandleFree"](self.handler))
            except Exception:
                pass

    def flagcxGetErrorString(self, result: flagcxResult_t) -> str:
        return self._funcs["flagcxGetErrorString"](result).decode("utf-8")

    def FLAGCX_CHECK(self, result: flagcxResult_t) -> None:
        if result != 0:
            error_str = self.flagcxGetErrorString(result)
            raise RuntimeError(f"FLAGCX error: {error_str}")

    def flagcxGetUniqueId(self) -> ctypes.POINTER(flagcxUniqueId):
        unique_id = ctypes.POINTER(flagcxUniqueId)()
        self.FLAGCX_CHECK(self._funcs["flagcxGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def flagcxCommInitRank(self, world_size: int, unique_id, rank: int) -> flagcxComm_t:
        comm = flagcxComm_t()
        self.FLAGCX_CHECK(
            self._funcs["flagcxCommInitRank"](ctypes.byref(comm), world_size, unique_id, rank)
        )
        return comm

    def flagcxBroadcast(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        root: int,
        comm: flagcxComm_t,
        stream: flagcxStream_t,
    ) -> None:
        self.FLAGCX_CHECK(
            self._funcs["flagcxBroadcast"](sendbuff, recvbuff, count, datatype, root, comm, stream)
        )

    def flagcxCommDestroy(self, comm: flagcxComm_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxCommDestroy"](comm))

    # Stream helpers via device handler
    def adaptor_stream_copy(self, torch_stream) -> flagcxStream_t:
        new_stream = flagcxStream_t()
        self.FLAGCX_CHECK(
            self.handler.contents.devHandle.contents.streamCopy(
                ctypes.byref(new_stream), ctypes.c_void_p(torch_stream.cuda_stream)
            )
        )
        return new_stream

    def adaptor_stream_free(self, stream: flagcxStream_t) -> None:
        self.FLAGCX_CHECK(self.handler.contents.devHandle.contents.streamFree(stream))


# =============================================================================
# PyFlagcxCommunicator — high-level communicator for weight sync
# =============================================================================


def _current_stream():
    """Get the current device stream in a device-agnostic way."""
    from verl.utils.device import get_torch_device

    device_mod = get_torch_device()
    return device_mod.current_stream()


class PyFlagcxCommunicator:
    """
    A standalone FlagCX communicator that bypasses torch.distributed.

    Uses StatelessProcessGroup (TCP) for rendezvous/unique_id distribution,
    then communicates via FlagCX C API directly.
    """

    def __init__(
        self,
        group,  # StatelessProcessGroup
        device: Union[str, int, torch.device],
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
        try:
            self.flagcx = FLAGCXLibrary(library_path)
        except Exception as e:
            logger.error(f"Failed to load FlagCX library: {e}")
            self.disabled = True
            return

        # Generate and distribute unique_id
        if self.rank == 0:
            self.unique_id = self.flagcx.flagcxGetUniqueId().contents
        else:
            self.unique_id = flagcxUniqueId()
        self.unique_id = group.broadcast_obj(self.unique_id, src=0)

        # Resolve device
        if isinstance(device, int):
            from verl.utils.device import get_device_name

            device_name = get_device_name()
            device = torch.device(f"{device_name}:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Initialize communicator under the correct device context
        if self.device.type == "musa":
            device_ctx = torch.musa.device(self.device)
        elif self.device.type == "cuda":
            device_ctx = torch.cuda.device(self.device)
        else:
            raise RuntimeError(f"Unsupported device type for FlagCX: {self.device.type}")

        with device_ctx:
            self.comm = self.flagcx.flagcxCommInitRank(
                self.world_size, ctypes.byref(self.unique_id), self.rank
            )
            # Warmup broadcast
            data = torch.zeros(1, device=self.device)
            self.broadcast(data, src=0)
            _current_stream().synchronize()
            del data

        logger.info(
            f"[PyFlagcxCommunicator] rank={self.rank}, world_size={self.world_size}, "
            f"device={self.device} initialized successfully"
        )

    def broadcast(self, tensor: torch.Tensor, src: int = 0, stream=None) -> None:
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"FlagCX communicator bound to {self.device}, but tensor is on {tensor.device}"
        )
        if stream is None:
            stream = _current_stream()
        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())
        flagcx_stream = self.flagcx.adaptor_stream_copy(stream)
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
