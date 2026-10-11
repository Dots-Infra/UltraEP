"""Python owns symmetric allocations; the extension consumes their peer pointers."""

from dataclasses import dataclass
from math import prod
from typing import Any

import torch
import torch.distributed._symmetric_memory as symm_mem


@dataclass
class SymmetricRegion:
    tensor: torch.Tensor
    handle: Any
    remote_ptrs: torch.Tensor


def symm_alloc(shape, *, dtype, device, group):
    shape = (shape,) if isinstance(shape, int) else shape
    numel = prod(shape)
    # Empty logical buffers still need an allocation for rendezvous and signal pads.
    storage = symm_mem.empty(
        max(numel, 16 // dtype.itemsize), dtype=dtype, device=device
    )
    handle = symm_mem.rendezvous(storage, group)
    pointers = handle.buffer_ptrs
    if len(pointers) != group.size() or any(ptr == 0 or ptr % 16 for ptr in pointers):
        raise RuntimeError(
            "Symmetric memory must map every NVLink peer with 16-byte alignment"
        )
    # The view retains the backing allocation, including any alignment padding.
    return SymmetricRegion(
        storage[:numel].view(shape),
        handle,
        torch.tensor(pointers, dtype=torch.int64, device="cpu"),
    )
