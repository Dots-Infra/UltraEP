"""Python owns symmetric allocations; the extension consumes their peer pointers."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed._symmetric_memory as symm_mem


@dataclass
class SymmetricRegion:
    storage: torch.Tensor
    handle: Any
    local: torch.Tensor
    remote_ptrs: torch.Tensor


def allocate_region(numel, *, dtype, device, group, local_shape=None):
    # Empty logical buffers still need an allocation for rendezvous and signal pads.
    element_size = torch.empty((), dtype=dtype).element_size()
    storage = symm_mem.empty(max(numel, 16 // element_size), dtype=dtype, device=device)
    handle = symm_mem.rendezvous(storage, group)
    pointers = handle.buffer_ptrs
    if len(pointers) != group.size() or any(ptr == 0 or ptr % 16 for ptr in pointers):
        raise RuntimeError(
            "Symmetric memory must map every NVLink peer with 16-byte alignment"
        )
    local = storage[:numel]
    if local_shape is not None:
        local = local.view(local_shape)
    return SymmetricRegion(
        storage, handle, local, torch.tensor(pointers, dtype=torch.int64, device="cpu")
    )
