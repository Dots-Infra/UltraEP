"""
Quota-aware placement solver tests for the single supported path:
quota placement v1 with the fastt oracle.
"""

import os
import sys
from typing import Tuple

import torch

try:
    import ultra_ep._C as _C
except ImportError:
    print("ERROR: Cannot import ultra_ep._C.", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_placement import EPConfig, validate_placement


def make_quota_solver_and_buffers(
    config: EPConfig,
) -> Tuple[
    object,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    num_global_logical = config.num_local_master * config.num_ranks
    num_local_physical = config.num_local_master + config.num_local_redundant
    num_global_physical = num_local_physical * config.num_ranks
    max_replicas_dim = config.num_ranks

    solver = _C.PlacementSolverQuota(
        num_global_logical,
        config.num_ranks,
        config.num_local_master,
        config.num_local_redundant,
        config.num_nvl_ranks,
        max_replicas_dim,
    )

    p2l = torch.full((num_global_physical,), -1, dtype=torch.int32, device="cuda")
    l2p = torch.full(
        (num_global_logical, max_replicas_dim), -1, dtype=torch.int32, device="cuda"
    )
    lcnts = torch.zeros(num_global_logical, dtype=torch.int32, device="cuda")
    quota = torch.zeros(
        (num_global_logical, max_replicas_dim), dtype=torch.int32, device="cuda"
    )
    quota_prefix = torch.zeros_like(quota)
    rank_quota_prefix = torch.zeros_like(quota)
    return solver, p2l, l2p, lcnts, quota, quota_prefix, rank_quota_prefix


def split_loads_per_rank(expert_loads: torch.Tensor, num_ranks: int) -> torch.Tensor:
    loads = expert_loads.to(torch.int32).cpu()
    num_experts = loads.numel()
    per_rank = torch.zeros(num_ranks, num_experts, dtype=torch.int32)
    base_weights = torch.arange(1, num_ranks + 1, dtype=torch.int64)

    for expert_idx in range(num_experts):
        load = int(loads[expert_idx].item())
        if load <= 0:
            continue
        weights = torch.roll(base_weights, shifts=expert_idx % num_ranks)
        scaled = weights * load
        parts = (scaled // weights.sum()).to(torch.int32)
        remainder = load - int(parts.sum().item())
        if remainder > 0:
            order = torch.argsort(-(scaled % weights.sum()))
            parts[order[:remainder]] += 1
        per_rank[:, expert_idx] = parts

    assert torch.equal(per_rank.sum(dim=0), loads)
    return per_rank


def solve_quota_case(
    config: EPConfig,
    expert_loads: torch.Tensor,
    locality_aware: bool = True,
    min_tokens_per_replica: int = 1,
    allow_zero_master_quota: bool = True,
    oracle_eps: float = 0.01,
    kernel_stage: int = 1,
):
    expert_loads = expert_loads.to(torch.int32)
    expert_loads_per_rank = split_loads_per_rank(expert_loads, config.num_ranks)
    solver, p2l, l2p, lcnts, quota, quota_prefix, rank_quota_prefix = (
        make_quota_solver_and_buffers(config)
    )

    solver.solve(
        expert_loads.cuda(),
        expert_loads_per_rank.cuda(),
        p2l,
        l2p,
        lcnts,
        quota,
        quota_prefix,
        rank_quota_prefix,
        1.0,
        min_tokens_per_replica,
        allow_zero_master_quota,
        locality_aware,
        oracle_eps,
        kernel_stage,
    )
    torch.cuda.synchronize()

    return {
        "expert_loads": expert_loads.cpu(),
        "expert_loads_per_rank": expert_loads_per_rank.cpu(),
        "p2l": p2l.cpu(),
        "l2p": l2p.cpu(),
        "lcnts": lcnts.cpu(),
        "quota": quota.cpu(),
        "quota_prefix": quota_prefix.cpu(),
        "rank_quota_prefix": rank_quota_prefix.cpu(),
    }


def validate_quota_state(
    config: EPConfig,
    expert_loads: torch.Tensor,
    expert_loads_per_rank: torch.Tensor,
    p2l: torch.Tensor,
    l2p: torch.Tensor,
    lcnts: torch.Tensor,
    quota: torch.Tensor,
    quota_prefix: torch.Tensor,
    rank_quota_prefix: torch.Tensor,
    my_rank: int = 0,
):
    validate_placement(expert_loads, p2l, l2p, lcnts, config)

    num_experts = expert_loads.numel()
    max_replicas = quota.size(1)
    for expert_idx in range(num_experts):
        replica_count = int(lcnts[expert_idx].item())
        local_total = int(expert_loads_per_rank[my_rank, expert_idx].item())
        total_load = int(expert_loads[expert_idx].item())

        assert replica_count >= 1
        assert int(quota[expert_idx, :replica_count].sum().item()) == total_load
        assert int(quota_prefix[expert_idx, replica_count - 1].item()) == total_load
        assert (
            int(rank_quota_prefix[expert_idx, replica_count - 1].item()) == local_total
        ), (
            f"expert {expert_idx}: rank_quota_prefix end="
            f"{rank_quota_prefix[expert_idx, replica_count - 1].item()} "
            f"!= local_total={local_total}"
        )

        prev_quota_prefix = 0
        prev_rank_prefix = 0
        for replica_idx in range(replica_count):
            curr_quota_prefix = int(quota_prefix[expert_idx, replica_idx].item())
            curr_rank_prefix = int(rank_quota_prefix[expert_idx, replica_idx].item())
            assert curr_quota_prefix >= prev_quota_prefix
            assert curr_rank_prefix >= prev_rank_prefix
            prev_quota_prefix = curr_quota_prefix
            prev_rank_prefix = curr_rank_prefix

        for replica_idx in range(replica_count, max_replicas):
            assert int(quota[expert_idx, replica_idx].item()) == 0
            assert int(quota_prefix[expert_idx, replica_idx].item()) == 0
            assert int(l2p[expert_idx, replica_idx].item()) == -1


def run_case(
    case_name: str,
    config: EPConfig,
    expert_loads: torch.Tensor,
    locality_aware: bool = True,
    min_tokens_per_replica: int = 1,
    allow_zero_master_quota: bool = True,
    oracle_eps: float = 0.01,
):
    outputs = solve_quota_case(
        config,
        expert_loads,
        locality_aware=locality_aware,
        min_tokens_per_replica=min_tokens_per_replica,
        allow_zero_master_quota=allow_zero_master_quota,
        oracle_eps=oracle_eps,
    )
    validate_quota_state(
        config,
        outputs["expert_loads"],
        outputs["expert_loads_per_rank"],
        outputs["p2l"],
        outputs["l2p"],
        outputs["lcnts"],
        outputs["quota"],
        outputs["quota_prefix"],
        outputs["rank_quota_prefix"],
        my_rank=0,
    )
    print(
        f"{case_name}: PASS (locality_aware={locality_aware}, oracle=fastt)",
        flush=True,
    )


def test_e2e_reroute_quota():
    """
    Validate that `rank_quota_prefix` partitions rank-local tokens exactly the
    same way as the quota reroute kernel's upper_bound-style dispatch.
    """
    config = EPConfig(
        num_ranks=4,
        num_local_master=4,
        num_local_redundant=2,
        num_nvl_ranks=4,
    )
    num_experts = config.num_ranks * config.num_local_master
    skewed_loads = torch.tensor(
        [512, 320, 256, 192] + [32] * (num_experts - 4), dtype=torch.int32
    )

    for case_name, locality_aware in (
        ("reroute_skewed_locality", True),
        ("reroute_skewed_no_locality", False),
    ):
        outputs = solve_quota_case(
            config,
            skewed_loads,
            locality_aware=locality_aware,
            oracle_eps=0.01,
        )
        lcnts = outputs["lcnts"]
        rank_quota_prefix = outputs["rank_quota_prefix"]
        expert_loads_per_rank = outputs["expert_loads_per_rank"]

        my_rank = 0
        my_loads = expert_loads_per_rank[my_rank]
        replica_token_counts: dict[tuple[int, int], int] = {}

        for expert_idx in range(num_experts):
            replica_count = int(lcnts[expert_idx].item())
            for replica_idx in range(replica_count):
                replica_token_counts[(expert_idx, replica_idx)] = 0

        for expert_idx in range(num_experts):
            replica_count = int(lcnts[expert_idx].item())
            for ordinal in range(int(my_loads[expert_idx].item())):
                replica_idx = 0
                for prefix_idx in range(replica_count):
                    if ordinal >= int(rank_quota_prefix[expert_idx, prefix_idx].item()):
                        replica_idx += 1
                replica_idx = min(replica_idx, max(replica_count - 1, 0))
                replica_token_counts[(expert_idx, replica_idx)] += 1

        for expert_idx in range(num_experts):
            replica_count = int(lcnts[expert_idx].item())
            total = 0
            for replica_idx in range(replica_count):
                curr_prefix = int(rank_quota_prefix[expert_idx, replica_idx].item())
                prev_prefix = (
                    int(rank_quota_prefix[expert_idx, replica_idx - 1].item())
                    if replica_idx > 0
                    else 0
                )
                expected = curr_prefix - prev_prefix
                actual = replica_token_counts[(expert_idx, replica_idx)]
                assert actual == expected, (
                    f"{case_name}: expert {expert_idx} replica {replica_idx} "
                    f"got {actual}, expected {expected}"
                )
                total += actual
            assert total == int(my_loads[expert_idx].item())

        print(f"{case_name}: PASS", flush=True)


def test_quota_kernel_stage_variants():
    config = EPConfig(
        num_ranks=4,
        num_local_master=4,
        num_local_redundant=2,
        num_nvl_ranks=4,
    )
    num_experts = config.num_ranks * config.num_local_master
    skewed_loads = torch.tensor(
        [512, 320, 256, 192] + [32] * (num_experts - 4), dtype=torch.int32
    )

    for kernel_stage in (0, 1):
        outputs = solve_quota_case(
            config,
            skewed_loads,
            locality_aware=True,
            oracle_eps=0.01,
            kernel_stage=kernel_stage,
        )
        validate_quota_state(
            config,
            outputs["expert_loads"],
            outputs["expert_loads_per_rank"],
            outputs["p2l"],
            outputs["l2p"],
            outputs["lcnts"],
            outputs["quota"],
            outputs["quota_prefix"],
            outputs["rank_quota_prefix"],
            my_rank=0,
        )


def main():
    if not torch.cuda.is_available():
        print("CUDA is required for test_placement_quota.py", file=sys.stderr)
        sys.exit(1)

    base_config = EPConfig(
        num_ranks=4,
        num_local_master=4,
        num_local_redundant=2,
        num_nvl_ranks=4,
    )
    num_experts = base_config.num_ranks * base_config.num_local_master

    run_case(
        "uniform",
        base_config,
        torch.full((num_experts,), 64, dtype=torch.int32),
        locality_aware=True,
    )
    skewed = torch.tensor(
        [512, 320, 256, 192] + [32] * (num_experts - 4), dtype=torch.int32
    )
    run_case("skewed", base_config, skewed, locality_aware=True, oracle_eps=0.01)
    run_case(
        "skewed_no_locality", base_config, skewed, locality_aware=False, oracle_eps=0.01
    )
    run_case(
        "single_hot",
        base_config,
        torch.tensor([1024] + [0] * (num_experts - 1), dtype=torch.int32),
        locality_aware=True,
    )
    run_case(
        "small_loads",
        base_config,
        torch.ones(num_experts, dtype=torch.int32),
        locality_aware=True,
    )

    no_redundant_config = EPConfig(
        num_ranks=4,
        num_local_master=4,
        num_local_redundant=0,
        num_nvl_ranks=4,
    )
    run_case(
        "no_redundant",
        no_redundant_config,
        torch.full(
            (no_redundant_config.num_ranks * no_redundant_config.num_local_master,),
            100,
            dtype=torch.int32,
        ),
        locality_aware=True,
    )

    test_e2e_reroute_quota()


if __name__ == "__main__":
    main()
