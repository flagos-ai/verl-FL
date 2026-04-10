# Copyright (c) 2026 BAAI. All rights reserved.
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
"""Auto-detect GPU vendor types across a Ray cluster for heterogeneous scheduling."""

import logging
from typing import Optional

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

logger = logging.getLogger(__name__)

# Map GPU device name keywords to canonical vendor names.
_VENDOR_KEYWORDS = {
    "nvidia": "nvidia",
    "tesla": "nvidia",
    "geforce": "nvidia",
    "quadro": "nvidia",
    "a100": "nvidia",
    "a800": "nvidia",
    "h100": "nvidia",
    "h800": "nvidia",
    "v100": "nvidia",
    "l40": "nvidia",
    "musa": "musa",
    "moore": "musa",
    "mt": "musa",
    "metax": "metax",
    "iluvatar": "iluvatar",
    "bi-v": "iluvatar",
    "ascend": "npu",
    "910b": "npu",
}


def _classify_vendor(device_name: str) -> str:
    """Classify a GPU device name string into a canonical vendor name."""
    name_lower = device_name.lower()
    for keyword, vendor in _VENDOR_KEYWORDS.items():
        if keyword in name_lower:
            return vendor
    return "unknown"


@ray.remote(num_cpus=0.01)
class _GpuProbeActor:
    """Lightweight actor dispatched to a specific node to detect GPU type."""

    def detect(self) -> dict:
        """Return node IP, GPU count, device name, and vendor."""
        ip = ray.util.get_node_ip_address()
        try:
            from verl.plugin.platform import get_platform

            platform = get_platform()
            if platform.is_available():
                device_name = platform.device_module.get_device_name(0)
                gpu_count = platform.device_count()
            else:
                device_name = "unknown"
                gpu_count = 0
        except Exception as e:
            logger.warning(f"GPU detection failed on node {ip}: {e}")
            device_name = "unknown"
            gpu_count = 0

        vendor = _classify_vendor(device_name)
        return {
            "ip": ip,
            "device_name": device_name,
            "gpu_count": gpu_count,
            "vendor": vendor,
        }


def inspect_cluster_gpu_types() -> dict[str, dict]:
    """Probe every GPU node in the Ray cluster and return per-node GPU info.

    Returns:
        dict mapping node_id -> {ip, device_name, gpu_count, vendor}

    Example output::

        {
            "node_abc123": {"ip": "10.0.0.1", "device_name": "NVIDIA A100", "gpu_count": 8, "vendor": "nvidia"},
            "node_def456": {"ip": "10.0.0.2", "device_name": "MTT S4000", "gpu_count": 8, "vendor": "musa"},
        }
    """
    gpu_nodes = [
        node
        for node in ray.nodes()
        if node["Alive"] and (node["Resources"].get("GPU", 0) > 0 or node["Resources"].get("NPU", 0) > 0)
    ]

    if not gpu_nodes:
        logger.warning("No alive GPU/NPU nodes found in the Ray cluster.")
        return {}

    # Dispatch a probe actor to each node
    probes = {}
    for node in gpu_nodes:
        node_id = node["NodeID"]
        actor = _GpuProbeActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote()
        probes[node_id] = actor

    # Collect results
    results = {}
    refs = {node_id: actor.detect.remote() for node_id, actor in probes.items()}
    for node_id, ref in refs.items():
        try:
            info = ray.get(ref, timeout=30)
            results[node_id] = info
            logger.info(
                f"Node {node_id} ({info['ip']}): {info['device_name']} x{info['gpu_count']} -> {info['vendor']}"
            )
        except Exception as e:
            logger.warning(f"Failed to probe node {node_id}: {e}")

    # Cleanup probe actors
    for actor in probes.values():
        ray.kill(actor)

    return results


def group_nodes_by_vendor(node_info: Optional[dict[str, dict]] = None) -> dict[str, list[dict]]:
    """Group cluster nodes by GPU vendor.

    Args:
        node_info: Output of inspect_cluster_gpu_types(). If None, will probe the cluster.

    Returns:
        dict mapping vendor -> list of {node_id, ip, gpu_count}

    Example::

        {
            "nvidia": [{"node_id": "abc", "ip": "10.0.0.1", "gpu_count": 8}],
            "musa":   [{"node_id": "def", "ip": "10.0.0.2", "gpu_count": 8}],
        }
    """
    if node_info is None:
        node_info = inspect_cluster_gpu_types()

    groups: dict[str, list[dict]] = {}
    for node_id, info in node_info.items():
        vendor = info["vendor"]
        if vendor not in groups:
            groups[vendor] = []
        groups[vendor].append(
            {
                "node_id": node_id,
                "ip": info["ip"],
                "gpu_count": info["gpu_count"],
            }
        )

    return groups
