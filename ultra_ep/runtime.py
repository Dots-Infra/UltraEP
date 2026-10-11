import os
from collections import Counter
from ctypes import byref

import pynvml
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from .util import print_rank_0
import ultra_ep._C as _C

_group = None
_symmetric_group = None
_nvl_domain_size = None
_manager_count = 0
_symmetric_groups = {}


def _nvlink_domain_key():
    """Identify the current GPU's fabric clique or fully connected local NVLink domain."""
    uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    pynvml.nvmlInit()
    try:
        device = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        info = pynvml.c_nvmlGpuFabricInfoV_t()
        try:
            pynvml.nvmlDeviceGetGpuFabricInfoV(device, byref(info))
        except (pynvml.NVMLError_NotSupported, pynvml.NVMLError_FunctionNotFound):
            pass
        if any(info.clusterUuid):
            if (
                info.state != pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
                or info.status != pynvml.NVML_SUCCESS
            ):
                raise RuntimeError("GPU fabric registration is not ready")
            return ("fabric", bytes(info.clusterUuid).hex(), info.cliqueId)

        # NVML enumerates physical GPUs independently of CUDA_VISIBLE_DEVICES.
        # Including our own UUID gives every GPU in a local clique the same key.
        peers = [uuid]
        for index in range(pynvml.nvmlDeviceGetCount()):
            peer = pynvml.nvmlDeviceGetHandleByIndex(index)
            peer_uuid = pynvml.nvmlDeviceGetUUID(peer)
            if (
                peer_uuid != uuid
                and pynvml.nvmlDeviceGetP2PStatus(
                    device, peer, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                )
                == pynvml.NVML_P2P_STATUS_OK
            ):
                peers.append(peer_uuid)
        return ("nvlink", tuple(sorted(peers)))
    finally:
        pynvml.nvmlShutdown()


def init_runtime(group: dist.ProcessGroup):
    """Use NCCL across EP ranks and symmetric memory within each NVLink domain."""
    global _group, _symmetric_group, _nvl_domain_size
    if _C.is_runtime_initialized():
        if group != _group:
            raise ValueError("All live UltraEP managers must share the same EP group")
        return _nvl_domain_size
    if group is None or group.size() <= 0 or group.rank() < 0:
        raise ValueError("UltraEP requires membership in a non-empty EP process group")
    if tuple(map(int, torch.__version__.split(".")[:2])) < (2, 11):
        raise RuntimeError(
            "UltraEP requires PyTorch >= 2.11 for NVLink-domain barriers"
        )
    if dist.get_backend(group) != "nccl":
        raise ValueError("UltraEP requires an NCCL EP process group")
    if symm_mem.get_backend("cuda") != "CUDA":
        symm_mem.set_backend("CUDA")

    domains = [None] * group.size()
    dist.all_gather_object(domains, _nvlink_domain_key(), group=group)
    counts = Counter(domains)
    detected = counts[domains[group.rank()]]
    domain_size = int(os.getenv("MAX_NUM_NVL_PEERS", detected))
    if (
        len(set(counts.values())) != 1
        or domain_size <= 0
        or detected % domain_size != 0
        or any(
            domains[i] != domains[i // domain_size * domain_size]
            for i in range(group.size())
        )
    ):
        raise ValueError(
            "NVLink domains must form equal contiguous EP-rank blocks; "
            f"MAX_NUM_NVL_PEERS={domain_size} must divide the detected size {detected}"
        )

    symmetric_group = group
    if domain_size < group.size():
        key = (group, domain_size)
        if key not in _symmetric_groups:
            ranks = dist.get_process_group_ranks(group)
            first = group.rank() // domain_size * domain_size
            _symmetric_groups[key] = dist.new_group(
                ranks=ranks[first : first + domain_size],
                backend="nccl",
                use_local_synchronization=True,
            )
        symmetric_group = _symmetric_groups[key]

    _C.init_runtime(group.rank(), group.size(), domain_size)
    _group, _symmetric_group, _nvl_domain_size = group, symmetric_group, domain_size
    print_rank_0(f"UltraEP: CUDA symmetric memory, NVLink domain size={domain_size}")
    return domain_size


def get_symmetric_group():
    return _symmetric_group


def retain_runtime():
    global _manager_count
    _manager_count += 1


def release_runtime():
    global _group, _symmetric_group, _nvl_domain_size, _manager_count
    _manager_count -= 1
    if _manager_count == 0:
        _C.destroy_runtime()
        _group = _symmetric_group = _nvl_domain_size = None
