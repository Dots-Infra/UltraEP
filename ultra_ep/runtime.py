import os
import torch.distributed as dist
from .util import print_rank_0

# noinspection PyUnresolvedReferences
import ultra_ep._C as _C

_group = None
_nvl_domain_size = None


def init_runtime(group: dist.ProcessGroup):
    global _group, _nvl_domain_size
    if _C.is_runtime_initialized():
        assert group == _group, f"All EP buffers should share the same process group"
        return _nvl_domain_size

    # * IMPORTANT: NVSHMEM environment variables
    # Disable NVLink SHArP to avoid cuMemMap failure
    os.environ["NVSHMEM_DISABLE_NVLS"] = "1"
    # NOTES: NVSHMEM initialization requires at least 256 MiB
    os.environ["NVSHMEM_CUMEM_GRANULARITY"] = f"{2 ** 29}"
    # Use primitive NVSHMEM for low-latency expert load all-reduce
    os.environ["NVSHMEM_DISABLE_NCCL"] = "1"

    # Synchronize NVSHMEM unique IDs
    root_unique_id = None
    if group.rank() == 0:
        root_unique_id = _C.get_local_nvshmem_unique_id(group.rank())
    nvshmem_unique_ids = [None] * group.size()
    dist.all_gather_object(nvshmem_unique_ids, root_unique_id, group)
    root_unique_id = nvshmem_unique_ids[0]

    # Support both MNNVL and RDMA by setting MAX_NVL_PEERS
    _ipc_manager = _C.IpcManager()
    print_rank_0(f"[INFO] Use MNNVL fabric: {_ipc_manager.is_fabric_supported()}")
    detected_ranks = _ipc_manager.detect_accessible_ranks(group)
    del _ipc_manager

    max_nvl_peers = os.getenv("MAX_NUM_NVL_PEERS")
    if max_nvl_peers is not None:
        max_nvl_peers = int(max_nvl_peers)
        if max_nvl_peers != detected_ranks:
            print_rank_0(
                f"[WARN] MAX_NUM_NVL_PEERS={max_nvl_peers} differs from detected value {detected_ranks}. Using environment variable."
            )
    else:
        max_nvl_peers = detected_ranks
        print_rank_0(
            f"[WARN] MAX_NUM_NVL_PEERS is not set. Using detected value {detected_ranks}."
        )

    # Initialize CPP runtime with NVSHMEM
    _C.init_runtime(group.rank(), group.size(), max_nvl_peers, root_unique_id)

    # Remember the EP group, which can not be changed anymore
    _group = group
    _nvl_domain_size = max_nvl_peers

    return max_nvl_peers
