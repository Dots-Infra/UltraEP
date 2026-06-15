import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def print_rank_0(message):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(message, flush=True)


def print_rank_k(message, rank):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == rank:
        print(message, flush=True)


def read_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def read_bool_env(name: str, default: bool) -> bool:
    value = read_env(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of: 0/1, true/false, yes/no, on/off")


def read_int_env(name: str, default: int) -> int:
    value = read_env(name)
    return default if value is None else int(value)


def read_float_env(name: str, default: float) -> float:
    value = read_env(name)
    return default if value is None else float(value)


def read_str_env(name: str, default: str) -> str:
    value = read_env(name)
    return default if value is None else value


if triton is not None:

    @triton.jit
    def _profile_physical_loads_from_quota_kernel(
        physical_to_logical,
        logical_to_physical,
        quota,
        post_loads,
        max_replicas: tl.constexpr,
        BLOCK_REPLICAS: tl.constexpr,
    ):
        physical_id = tl.program_id(axis=0)
        logical_id = tl.load(physical_to_logical + physical_id)
        valid_logical = logical_id >= 0

        replica_offsets = tl.arange(0, BLOCK_REPLICAS)
        replica_mask = replica_offsets < max_replicas
        safe_logical_id = tl.where(valid_logical, logical_id, 0)
        row_offsets = safe_logical_id * max_replicas + replica_offsets

        physical_slots = tl.load(
            logical_to_physical + row_offsets,
            mask=valid_logical & replica_mask,
            other=-1,
        )
        matched = physical_slots == physical_id
        load_values = tl.load(
            quota + row_offsets,
            mask=valid_logical & replica_mask & matched,
            other=0,
        )
        physical_load = tl.sum(tl.where(matched, load_values, 0), axis=0)
        tl.store(post_loads + physical_id, physical_load)


def _require_triton():
    if triton is None:
        raise ImportError(
            "Triton is required for fused UltraEP load profiling. "
            "Install triton or call _profile_physical_loads_from_quota(..., fused=False)."
        )


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def profile_physical_loads_from_quota_triton(
    physical_to_logical: torch.Tensor,
    logical_to_physical: torch.Tensor,
    quota: torch.Tensor,
) -> torch.Tensor:
    _require_triton()
    assert triton is not None

    if physical_to_logical.dim() != 1:
        raise ValueError("physical_to_logical must be a 1D tensor")
    if logical_to_physical.dim() != 2:
        raise ValueError("logical_to_physical must be a 2D tensor")
    if quota.shape != logical_to_physical.shape:
        raise ValueError("quota must have the same shape as logical_to_physical")
    if (
        physical_to_logical.dtype != torch.int32
        or logical_to_physical.dtype != torch.int32
        or quota.dtype != torch.int32
    ):
        raise TypeError("profile load tensors must be torch.int32")
    if (
        not physical_to_logical.is_cuda
        or not logical_to_physical.is_cuda
        or not quota.is_cuda
    ):
        raise ValueError("profile load tensors must be CUDA tensors")
    if (
        physical_to_logical.device != logical_to_physical.device
        or physical_to_logical.device != quota.device
    ):
        raise ValueError("profile load tensors must be on the same CUDA device")
    if (
        not physical_to_logical.is_contiguous()
        or not logical_to_physical.is_contiguous()
        or not quota.is_contiguous()
    ):
        raise ValueError("profile load tensors must be contiguous")

    num_physical = physical_to_logical.numel()
    post_loads = torch.empty_like(physical_to_logical)
    if num_physical == 0:
        return post_loads

    max_replicas = logical_to_physical.size(1)
    block_replicas = _next_power_of_2(max_replicas)
    num_warps = 1 if block_replicas <= 64 else 4
    _profile_physical_loads_from_quota_kernel[(num_physical,)](
        physical_to_logical,
        logical_to_physical,
        quota,
        post_loads,
        max_replicas,
        BLOCK_REPLICAS=block_replicas,
        num_warps=num_warps,
    )
    return post_loads
