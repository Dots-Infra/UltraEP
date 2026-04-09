"""
Quota-aware placement solver tests.

This script exercises the CUDA-facing ``PlacementSolverQuota`` binding directly
and validates both the legacy placement invariants and the new quota-specific
state tensors.
"""

import os
import sys

import torch

try:
    import ultra_ep._C as _C
except ImportError:
    print("ERROR: Cannot import ultra_ep._C.", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_placement import EPConfig, validate_placement


def make_solver_and_buffers(config: EPConfig):
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
    L = expert_loads.numel()
    result = torch.zeros(num_ranks, L, dtype=torch.int32)
    base_weights = torch.arange(1, num_ranks + 1, dtype=torch.int64)
    for l in range(L):
        load = int(expert_loads[l].item())
        if load <= 0:
            continue
        weights = torch.roll(base_weights, shifts=l % num_ranks)
        scaled = weights * load
        parts = (scaled // weights.sum()).to(torch.int32)
        remainder = load - int(parts.sum().item())
        if remainder > 0:
            order = torch.argsort(-(scaled % weights.sum()))
            parts[order[:remainder]] += 1
        result[:, l] = parts
    assert torch.equal(result.sum(dim=0), expert_loads.cpu())
    return result


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

    L = expert_loads.numel()
    R = quota.size(1)
    for l in range(L):
        C = int(lcnts[l].item())
        assert C >= 1
        assert int(quota[l, :C].sum().item()) == int(expert_loads[l].item())
        if C > 0:
            assert int(quota_prefix[l, C - 1].item()) == int(expert_loads[l].item())
        for j in range(1, C):
            assert int(quota_prefix[l, j].item()) >= int(quota_prefix[l, j - 1].item())
        for j in range(C, R):
            assert int(quota[l, j].item()) == 0
            assert int(quota_prefix[l, j].item()) == 0
            assert int(l2p[l, j].item()) == -1

        local_total = int(expert_loads_per_rank[my_rank, l].item())
        if C > 0:
            assert int(rank_quota_prefix[l, C - 1].item()) == local_total, (
                f"expert {l}: rank_quota_prefix[C-1]={rank_quota_prefix[l, C-1].item()} "
                f"!= local_total={local_total} (rank={my_rank})"
            )
        prev = 0
        for j in range(C):
            curr = int(rank_quota_prefix[l, j].item())
            assert curr >= prev, (
                f"expert {l}: rank_quota_prefix not monotonic at j={j} "
                f"(prev={prev}, curr={curr}, rank={my_rank})"
            )
            prev = curr


def run_case(
    config: EPConfig,
    expert_loads: torch.Tensor,
    locality_aware: bool,
    v1_oracle_eps: float = 0.01,
    v1_kernel_stage: int = 0,
):
    if v1_kernel_stage not in (0, 1):
        raise ValueError("v1_kernel_stage supports only {0, 1}; stage 2/3 has been removed")
    expert_loads = expert_loads.to(torch.int32)
    expert_loads_per_rank = split_loads_per_rank(expert_loads, config.num_ranks)
    solver, p2l, l2p, lcnts, quota, quota_prefix, rank_quota_prefix = (
        make_solver_and_buffers(config)
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
        1,
        True,
        locality_aware,
        v1_oracle_eps,
        v1_kernel_stage,
    )
    torch.cuda.synchronize()

    # In standalone test (no distributed runtime), solver uses rank 0
    validate_quota_state(
        config,
        expert_loads.cpu(),
        expert_loads_per_rank.cpu(),
        p2l.cpu(),
        l2p.cpu(),
        lcnts.cpu(),
        quota.cpu(),
        quota_prefix.cpu(),
        rank_quota_prefix.cpu(),
        my_rank=0,
    )


def test_e2e_reroute_quota():
    """
    End-to-end test: solver output + Python reference reroute scatter (QUOTA_MODE).

    Simulates the branchless upper_bound logic of reroute_forward_scatter_kernel
    in Python to verify that rank_quota_prefix correctly partitions each rank's
    tokens among physical replicas.
    """
    config = EPConfig(
        num_ranks=4,
        num_local_master=4,
        num_local_redundant=2,
        num_nvl_ranks=4,
    )
    L = config.num_ranks * config.num_local_master
    max_replicas_dim = config.num_ranks

    cases = [
        ("uniform", torch.full((L,), 64, dtype=torch.int32), True),
        (
            "skewed",
            torch.tensor(
                [512, 320, 256, 192] + [32] * (L - 4), dtype=torch.int32
            ),
            True,
        ),
        (
            "skewed_no_locality",
            torch.tensor(
                [512, 320, 256, 192] + [32] * (L - 4), dtype=torch.int32
            ),
            False,
        ),
    ]

    for test_name, expert_loads_cpu, locality_aware in cases:
        expert_loads_per_rank = split_loads_per_rank(expert_loads_cpu, config.num_ranks)

        solver, p2l, l2p, lcnts, quota, quota_prefix, rank_quota_prefix = (
            make_solver_and_buffers(config)
        )
        solver.solve(
            expert_loads_cpu.cuda(),
            expert_loads_per_rank.cuda(),
            p2l,
            l2p,
            lcnts,
            quota,
            quota_prefix,
            rank_quota_prefix,
            1.0,
            1,
            True,
            locality_aware,
            0.01,  # v1_oracle_eps
        )
        torch.cuda.synchronize()

        l2p_cpu = l2p.cpu()
        lcnts_cpu = lcnts.cpu()
        rqp_cpu = rank_quota_prefix.cpu()

        my_rank = 0
        my_loads = expert_loads_per_rank[my_rank]

        # Simulate QUOTA_MODE reroute scatter (Python reference).
        # For each expert l, iterate over all ordinal values [0, my_loads[l])
        # and apply the branchless upper_bound to determine which replica
        # receives the token.
        replica_token_counts: dict[tuple[int, int], int] = {}
        for l in range(L):
            C = int(lcnts_cpu[l].item())
            for j in range(C):
                replica_token_counts[(l, j)] = 0

        for l in range(L):
            load_l = int(my_loads[l].item())
            C = int(lcnts_cpu[l].item())
            for ordinal in range(load_l):
                # Branchless upper_bound: count how many prefix values <= ordinal
                replica_idx = 0
                for j in range(C):
                    if ordinal >= int(rqp_cpu[l, j].item()):
                        replica_idx += 1
                replica_idx = min(replica_idx, max(C - 1, 0))
                replica_token_counts[(l, replica_idx)] += 1

        # Verify per-replica counts match rank_quota_prefix differences
        for l in range(L):
            C = int(lcnts_cpu[l].item())
            total = 0
            for j in range(C):
                prefix_j = int(rqp_cpu[l, j].item())
                prefix_prev = int(rqp_cpu[l, j - 1].item()) if j > 0 else 0
                expected = prefix_j - prefix_prev
                actual = replica_token_counts[(l, j)]
                assert actual == expected, (
                    f"[{test_name}] expert {l} replica {j}: "
                    f"got {actual} tokens, expected {expected} "
                    f"(rqp[{j}]={prefix_j}, rqp_prev={prefix_prev})"
                )
                total += actual
            assert total == int(my_loads[l].item()), (
                f"[{test_name}] expert {l}: total {total} "
                f"!= my_loads {int(my_loads[l].item())}"
            )

        print(f"e2e reroute quota ({test_name}): PASS", flush=True)


def main():
    if not torch.cuda.is_available():
        print("CUDA is required for test_placement_quota.py", file=sys.stderr)
        sys.exit(1)

    config = EPConfig(
        num_ranks=4,
        num_local_master=4,
        num_local_redundant=2,
        num_nvl_ranks=4,
    )
    L = config.num_ranks * config.num_local_master

    cases = [
        torch.full((L,), 64, dtype=torch.int32),
        torch.tensor([512, 320, 256, 192] + [32] * (L - 4), dtype=torch.int32),
        torch.tensor(
            [256 if i % 3 == 0 else 48 if i % 3 == 1 else 8 for i in range(L)],
            dtype=torch.int32,
        ),
    ]

    for locality_aware in (True, False):
        for idx, expert_loads in enumerate(cases):
            run_case(config, expert_loads, locality_aware=locality_aware)
            print(
                f"case {idx}, locality_aware={locality_aware}: PASS",
                flush=True,
            )

    # fast_t path sanity check
    run_case(
        config,
        cases[1],  # skewed case
        locality_aware=True,
        v1_oracle_eps=0.01,
    )
    print("fast_t sanity: PASS", flush=True)

    for stage in (1,):
        run_case(
            config,
            cases[1],  # skewed case
            locality_aware=True,
            v1_oracle_eps=0.01,
            v1_kernel_stage=stage,
        )
        print(f"v4 stage={stage} sanity: PASS", flush=True)

    # ---- Edge cases ----

    # Edge case: all zero loads
    zero_loads = torch.zeros(L, dtype=torch.int32)
    run_case(config, zero_loads, locality_aware=True)
    print("edge case (zero loads): PASS", flush=True)

    # Edge case: single expert has all load
    single_load = torch.zeros(L, dtype=torch.int32)
    single_load[0] = 1024
    run_case(config, single_load, locality_aware=True)
    print("edge case (single expert all load): PASS", flush=True)

    # Edge case: very small loads (min_tokens_per_replica boundary)
    small_loads = torch.ones(L, dtype=torch.int32)
    run_case(config, small_loads, locality_aware=True)
    print("edge case (load=1 per expert): PASS", flush=True)

    # Edge case: C=1 (no redundant experts)
    config_no_redundant = EPConfig(
        num_ranks=4,
        num_local_master=4,
        num_local_redundant=0,
        num_nvl_ranks=4,
    )
    L_nr = config_no_redundant.num_ranks * config_no_redundant.num_local_master
    nr_loads = torch.full((L_nr,), 100, dtype=torch.int32)
    run_case(config_no_redundant, nr_loads, locality_aware=True)
    print("edge case (C=1, no redundant): PASS", flush=True)

    # Edge case: num_local_redundant=1
    config_one_redundant = EPConfig(
        num_ranks=4,
        num_local_master=4,
        num_local_redundant=1,
        num_nvl_ranks=4,
    )
    L_1r = config_one_redundant.num_ranks * config_one_redundant.num_local_master
    one_r_loads = torch.tensor(
        [500, 200, 100, 50] + [30] * (L_1r - 4), dtype=torch.int32
    )
    run_case(config_one_redundant, one_r_loads, locality_aware=True)
    print("edge case (1 redundant): PASS", flush=True)

    # ---- End-to-end reroute quota test ----
    test_e2e_reroute_quota()


if __name__ == "__main__":
    main()
