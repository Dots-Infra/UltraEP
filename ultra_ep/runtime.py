import os
import torch.distributed as dist
from .util import print_rank_0

# noinspection PyUnresolvedReferences
import ultra_ep._C as _C

_group = None


def init_runtime(group: dist.ProcessGroup):
    global _group
    if _C.is_runtime_initialized():
        assert group == _group, f"All EP buffers should share the same process group"
        return

    # * IMPORTANT: NVSHMEM environment variables
    # Disable NVLink SHArP to avoid cuMemMap failure
    os.environ["NVSHMEM_DISABLE_NVLS"] = "1"
    # NOTES: NVSHMEM initialization requires at least 256 MiB
    os.environ["NVSHMEM_CUMEM_GRANULARITY"] = f"{2 ** 29}"

    # Synchronize NVSHMEM unique IDs
    root_unique_id = None
    if group.rank() == 0:
        root_unique_id = _C.get_local_nvshmem_unique_id(group.rank())
    nvshmem_unique_ids = [None] * group.size()
    dist.all_gather_object(nvshmem_unique_ids, root_unique_id, group)
    root_unique_id = nvshmem_unique_ids[0]

    # Support both MNNVL and RDMA by setting MAX_NVL_PEERS
    allocator = _C.RemoteMemAllocator()
    print_rank_0(f"[INFO] Use MNNVL fabric: {allocator.is_fabric_supported()}")
    detected_ranks = allocator.detect_accessible_ranks(group)
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

    # Initialize CPP runtime
    _C.init_runtime(group.rank(), group.size(), max_nvl_peers, root_unique_id)

    # Remember the EP group, which can not be changed anymore
    _group = group


def sync_ipc_handles(runtime):
    global _group
    assert (
        _C.is_runtime_initialized()
    ), "Runtime must be initialized before syncing IPC handles"
    assert _group is not None

    # Synchronize IPC handles
    all_gathered_weight_handles = [None] * _group.size()
    all_gathered_grad_handles = [None] * _group.size()
    dist.all_gather_object(
        all_gathered_weight_handles, runtime.get_local_weight_ipc_handle(), _group
    )
    dist.all_gather_object(
        all_gathered_grad_handles, runtime.get_local_grad_ipc_handle(), _group
    )

    runtime.sync_global_ipc_handles(
        all_gathered_weight_handles, all_gathered_grad_handles
    )
