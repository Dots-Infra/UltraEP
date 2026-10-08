"""Small Python-side adapter for PyTorch Symmetric Memory.

UltraEP's CUDA kernels only need the local allocation and a table of mapped
peer addresses.  Keeping allocation/rendezvous here avoids depending on
PyTorch's private Symmetric Memory C++ headers or ABI from the extension.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Sequence

import torch
import torch.distributed._symmetric_memory as symm_mem


_ALIGNMENT_BYTES = 16


def _pointer_value(value: Any) -> int:
    """Normalize a handle buffer pointer to a Python integer."""

    if value is None:
        return 0
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise RuntimeError("Symmetric Memory returned a non-scalar peer pointer")
        return int(value.item())
    return int(value)


@dataclass
class SymmetricRegion:
    """One symmetric allocation and its group-local pointer table."""

    storage: torch.Tensor
    handle: Any
    local: torch.Tensor
    remote_ptrs: torch.Tensor


def _aligned_allocation_size(numel: int, element_size: int) -> int:
    requested_bytes = max(int(numel), 0) * int(element_size)
    # Symmetric Memory does not need a zero-sized allocation.  Keep a small
    # aligned backing allocation for empty logical buffers (e.g. inference
    # gradients or a model with no redundant experts).
    return max(
        _ALIGNMENT_BYTES,
        (requested_bytes + _ALIGNMENT_BYTES - 1) // _ALIGNMENT_BYTES * _ALIGNMENT_BYTES,
    )


def allocate_region(
    numel: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    group: Any,
    local_shape: Sequence[int] | None = None,
) -> SymmetricRegion:
    """Allocate/rendezvous a symmetric 1-D region and return a logical view.

    All ranks in ``group`` must pass the same allocation size, dtype and shape.
    ``local_shape`` may describe a zero-length logical view backed by the
    aligned dummy allocation.
    """

    numel = int(numel)
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel}")
    element_size = torch.empty((), dtype=dtype).element_size()
    storage_numel = _aligned_allocation_size(numel, element_size) // element_size
    if local_shape is not None:
        shape = tuple(int(dim) for dim in local_shape)
        if any(dim < 0 for dim in shape):
            raise ValueError(
                f"local_shape must contain non-negative dimensions, got {shape}"
            )
        logical_numel = 1
        for dim in shape:
            logical_numel *= dim
        if logical_numel != numel:
            raise ValueError(
                f"local_shape={shape} has {logical_numel} elements, expected {numel}"
            )
    storage = symm_mem.empty(storage_numel, dtype=dtype, device=device)
    handle = symm_mem.rendezvous(storage, group)

    pointers = [_pointer_value(ptr) for ptr in handle.buffer_ptrs]
    expected = group.size()
    if int(handle.world_size) != expected:
        raise RuntimeError(
            f"Symmetric Memory handle world size {handle.world_size} does not match NVLink group size {expected}"
        )
    if len(pointers) != expected:
        raise RuntimeError(
            f"Symmetric Memory returned {len(pointers)} peer pointers for group size {expected}"
        )
    if any(ptr == 0 for ptr in pointers):
        raise RuntimeError(
            "Symmetric Memory did not map every NVLink peer; all ranks in the symmetric group must be in the same NVLink domain"
        )
    if any(ptr % _ALIGNMENT_BYTES != 0 for ptr in pointers):
        raise RuntimeError(
            f"Symmetric Memory returned a peer pointer that is not {_ALIGNMENT_BYTES}-byte aligned"
        )
    if storage.data_ptr() % _ALIGNMENT_BYTES != 0:
        raise RuntimeError(
            f"Symmetric Memory allocation is not {_ALIGNMENT_BYTES}-byte aligned: "
            f"0x{storage.data_ptr():x}"
        )

    logical = storage.narrow(0, 0, numel)
    if local_shape is not None:
        logical = logical.view(shape)
    remote_ptrs = torch.tensor(pointers, dtype=torch.int64, device="cpu")
    return SymmetricRegion(storage, handle, logical, remote_ptrs)


def select_backend() -> str:
    """Use CUDA by default; NCCL requires an explicit opt-in before allocation."""
    requested = (
        os.getenv("ULTRA_EP_SYMMETRIC_MEMORY_BACKEND", "").strip().upper() or "CUDA"
    )
    if requested not in {"CUDA", "NCCL"}:
        raise ValueError(
            f"ULTRA_EP_SYMMETRIC_MEMORY_BACKEND must be CUDA or NCCL; got {requested!r}"
        )
    # Some Torch versions register the default allocator separately from the
    # set_backend() candidates. Keep an already selected backend as-is.
    if symm_mem.get_backend("cuda") != requested:
        symm_mem.set_backend(requested)
    return requested
