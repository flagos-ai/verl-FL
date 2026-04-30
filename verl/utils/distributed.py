# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for distributed training."""

import ctypes
import os
from datetime import timedelta

import ray
import torch.distributed

from verl.utils.device import get_dist_backend, get_nccl_backend, get_torch_device, is_npu_available


def init_device_mesh_hetero(device_type, mesh_shape, mesh_dim_names=None):
    """FlagCX-compatible init_device_mesh that supports heterogeneous device types.

    In heterogeneous environments (e.g. MUSA x NVIDIA), different ranks have
    different device_type (musa vs cuda). Standard init_device_mesh requires
    all ranks to pass the same device_type, which fails. This function uses
    dist.new_group + DeviceMesh.from_group to bypass that restriction when
    FlagCX is the communication backend.
    """
    if get_nccl_backend() != "flagcx":
        from torch.distributed.device_mesh import init_device_mesh

        return init_device_mesh(device_type, mesh_shape, mesh_dim_names=mesh_dim_names)

    import itertools
    import math

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh

    rank = dist.get_rank()
    backend = get_dist_backend()
    total = math.prod(mesh_shape)
    mesh_tensor = torch.arange(total).reshape(mesh_shape)
    ndim = len(mesh_shape)

    if ndim == 1:
        # If the group covers all ranks, reuse the default pg to avoid
        # creating a redundant FlagCX communicator (which can fail and
        # corrupt the default pg state on some platforms like MUSA).
        if total == dist.get_world_size():
            group = dist.distributed_c10d._get_default_group()
        else:
            group = dist.new_group(ranks=list(range(total)), backend=backend)
        return DeviceMesh.from_group(group, device_type, mesh_tensor, mesh_dim_names=mesh_dim_names)

    # Multi-dim: create subgroups for each dimension
    my_groups = []
    for dim in range(ndim):
        other_dims = [d for d in range(ndim) if d != dim]
        other_ranges = [range(mesh_shape[d]) for d in other_dims]

        my_group = None
        for combo in itertools.product(*other_ranges):
            idx = [None] * ndim
            for i, d in enumerate(other_dims):
                idx[d] = combo[i]
            idx[dim] = slice(None)

            ranks = mesh_tensor[tuple(idx)].flatten().tolist()
            g = dist.new_group(ranks=ranks, backend=backend)
            if rank in ranks:
                my_group = g

        my_groups.append(my_group)

    return DeviceMesh.from_group(my_groups, device_type, mesh_tensor, mesh_dim_names=mesh_dim_names)


def set_numa_affinity():
    if is_npu_available:
        # TODO (FightingZhen) libnuma.so is not available in e2e_ascend CI image, remove this code after image update.
        return

    initialized = False
    try:
        libnuma = ctypes.CDLL("libnuma.so")
        if libnuma.numa_available() < 0:
            return

        import pynvml

        pynvml.nvmlInit()
        initialized = True
        device_name = "NPU" if is_npu_available else "GPU"
        local_rank = int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        pynvml.nvmlDeviceSetCpuAffinity(handle)
    except ImportError:
        print("Warning: pynvml not available, skipping NUMA affinity setup")
    except Exception as e:
        print(f"Warning: Failed to set NUMA affinity: {e}")
    finally:
        if initialized:
            pynvml.nvmlShutdown()


def initialize_global_process_group(timeout_second=36000):
    torch.distributed.init_process_group(
        get_dist_backend(),
        timeout=timedelta(seconds=timeout_second),
        init_method=os.environ.get("DIST_INIT_METHOD", None),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        get_torch_device().set_device(local_rank)
    return local_rank, rank, world_size


def destroy_global_process_group():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def initialize_global_process_group_ray(timeout_second=None):
    # in current ray environment, LOCAL_RANK is always zero.

    import torch.distributed

    timeout = timedelta(seconds=timeout_second) if timeout_second is not None else None

    if not torch.distributed.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.distributed.init_process_group(
            backend=get_dist_backend(),
            rank=rank,
            world_size=world_size,
            timeout=timeout,
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )
