# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import torch
import torch.distributed as dist

from verl.utils.device import get_nccl_backend, is_npu_available


class _TorchDistBroadcastGroup:
    """Thin wrapper around a ``torch.distributed`` process group that exposes
    the same ``broadcast(tensor, src, stream)`` interface used by vLLM's
    ``PyNcclCommunicator`` / ``PyHcclCommunicator``.

    Used in FlagCX environments to perform device-side broadcast directly
    without CPU round-trips.  Calls the PG's low-level ``broadcast`` method
    directly to avoid the global-registry check in ``dist.broadcast``.
    """

    def __init__(self, group):
        self.group = group

    def broadcast(self, tensor, src=0, stream=None):
        opts = dist.BroadcastOptions()
        opts.rootRank = src
        opts.rootTensor = 0
        work = self.group.broadcast([tensor], opts)
        work.wait()


def _create_flagcx_weight_sync_group(master_address, master_port, rank, world_size, device):
    """Create a weight-sync process group for FlagCX environments.

    Uses a standalone ``TCPStore`` for rendezvous and creates a FlagCX
    process group via ``_new_process_group_helper``, which correctly
    constructs the ``_DistributedBackendOptions`` and backend-specific
    Options required by the FlagCX plugin.
    """
    from datetime import timedelta

    from verl.utils.device import get_dist_backend

    store = dist.TCPStore(
        host_name=master_address,
        port=master_port,
        world_size=world_size,
        is_master=(rank == 0),
    )
    prefix_store = dist.PrefixStore(f"weight_sync_{master_port}", store)

    backend = get_dist_backend()
    # _new_process_group_helper returns (ProcessGroup, PrefixStore) tuple.
    pg, _ = dist.distributed_c10d._new_process_group_helper(
        group_size=world_size,
        group_rank=rank,
        global_ranks_in_group=list(range(world_size)),
        backend=backend,
        store=prefix_store,
        group_name=f"weight_sync_{master_port}",
        timeout=timedelta(seconds=1800),
        device_id=torch.device(device),
    )
    return _TorchDistBroadcastGroup(pg)


def vllm_stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    comm_backend = get_nccl_backend()

    # FlagCX: create a standalone FlagCX process group for device-side
    # broadcast, bypassing vLLM's PyNcclCommunicator which needs native NCCL.
    if comm_backend not in ("nccl", "gloo", "hccl"):
        return _create_flagcx_weight_sync_group(master_address, master_port, rank, world_size, device)

    # NOTE: If it is necessary to support weight synchronization with the sglang backend in the future,
    # the following can be used:
    # from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    # from sglang.srt.distributed.utils import statelessprocessgroup
    if is_npu_available:
        from vllm_ascend.distributed.device_communicators.pyhccl import (
            PyHcclCommunicator as PyNcclCommunicator,
        )
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl
