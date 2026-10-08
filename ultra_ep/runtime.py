import os

import torch
import torch.distributed as dist

from .symmetric_memory import select_backend
from .util import print_rank_0
import ultra_ep._C as _C

_group = None
_symmetric_group = None
_nvl_domain_size = None
_manager_count = 0
_symmetric_groups = {}


def init_runtime(group: dist.ProcessGroup):
    """Keep placement on the EP group and peer mappings within each NVLink domain.

    Equally sized domains occupy contiguous blocks of EP-group ranks.
    MAX_NUM_NVL_PEERS can restrict the detected domain size. All EP ranks must
    construct and destroy managers in the same order.
    """
    global _group, _symmetric_group, _nvl_domain_size
    if _C.is_runtime_initialized():
        if group != _group:
            raise ValueError("All live UltraEP managers must share the same EP group")
        return _nvl_domain_size
    if group is None or group.size() <= 0 or group.rank() < 0:
        raise ValueError("UltraEP requires membership in a non-empty EP process group")

    backend = select_backend()
    ipc_manager = _C.IpcManager()
    detected = ipc_manager.detect_accessible_ranks(group)
    del ipc_manager
    domain_size = int(os.getenv("MAX_NUM_NVL_PEERS", detected))
    if domain_size <= 0 or domain_size > detected or group.size() % domain_size:
        raise ValueError(
            f"MAX_NUM_NVL_PEERS={domain_size} must divide EP size {group.size()} "
            f"and be between 1 and the detected peer count {detected}"
        )

    # CUDA Symmetric Memory can use the existing EP group when it is one
    # NVLink domain. Only the opt-in NCCL allocator requires an eager comm.
    symmetric_group = group
    if domain_size < group.size() or backend == "NCCL":
        key = (group, domain_size, backend)
        symmetric_group = _symmetric_groups.get(key)
        if symmetric_group is None:
            ranks = dist.get_process_group_ranks(group)
            first = group.rank() // domain_size * domain_size
            symmetric_group = dist.new_group(
                ranks=ranks[first : first + domain_size],
                backend="nccl",
                use_local_synchronization=True,
                device_id=(
                    torch.device("cuda", torch.cuda.current_device())
                    if backend == "NCCL"
                    else None
                ),
            )
            # Cache subgroups until global teardown: recreating the same rank
            # hash can reuse stale NCCL bootstrap keys in some Torch versions.
            _symmetric_groups[key] = symmetric_group
    _C.init_runtime(group.rank(), group.size(), domain_size)
    _group = group
    _symmetric_group = symmetric_group
    _nvl_domain_size = domain_size
    print_rank_0(
        f"UltraEP: Torch symmetric memory backend={backend}, NVLink domain size={domain_size}"
    )
    return domain_size


def get_symmetric_group():
    return _symmetric_group


def retain_runtime():
    global _manager_count
    _manager_count += 1


def release_runtime():
    """Reset native topology after the last manager releases its allocations."""
    global _group, _symmetric_group, _nvl_domain_size, _manager_count
    _manager_count -= 1
    if _manager_count == 0:
        _C.destroy_runtime()
        _group = _symmetric_group = _nvl_domain_size = None
