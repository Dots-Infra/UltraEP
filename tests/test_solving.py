import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import ultra_ep._C as _C

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    bench,
    bench_kineto,
    generate_loads_per_rank_zipf,
    max_mean,
    parse_csv_strings,
    summarize_vector,
)


def solve_once(loads_per_rank, args, legacy: bool):
    global_loads = loads_per_rank.sum(dim=0, dtype=torch.int32)
    return _C.solve_placement_for_test(
        global_loads,
        loads_per_rank,
        args.num_ranks,
        args.num_local_master,
        args.num_redundant_experts_per_rank,
        args.nvl_domain_size,
        legacy,
        args.balance_threshold,
        args.quota_min_tokens_per_replica,
        args.quota_allow_zero_master_quota,
        args.quota_locality_aware,
        args.quota_oracle_eps,
        args.quota_kernel_stage,
    )


def assert_legal(p2l, l2p, lcnts, args):
    num_local_physical = args.num_local_master + args.num_redundant_experts_per_rank
    logical_ids = torch.arange(args.num_experts, device=p2l.device)
    master_rank = logical_ids // args.num_local_master
    valid = l2p >= 0
    phys = l2p.clamp_min(0)
    phys_rank = phys // num_local_physical
    phys_local = phys % num_local_physical

    expected_master = master_rank * num_local_physical + logical_ids % args.num_local_master
    assert torch.equal(l2p[:, 0], expected_master)
    assert torch.equal(lcnts, valid.sum(dim=1).to(torch.int32))

    same_domain = (
        phys_rank // args.nvl_domain_size
        == master_rank.unsqueeze(1) // args.nvl_domain_size
    )
    assert bool((same_domain | ~valid).all().item()), "replica crosses NVL domain"

    rank_one_hot = torch.nn.functional.one_hot(
        phys_rank.clamp(0, args.num_ranks - 1).to(torch.int64), num_classes=args.num_ranks
    ).to(torch.int32)
    per_rank_copies = (rank_one_hot * valid.unsqueeze(-1)).sum(dim=1)
    assert bool((per_rank_copies <= 1).all().item()), "duplicate logical expert on a rank"

    is_redundant_slot = phys_local >= args.num_local_master
    assert bool(((is_redundant_slot | ~valid) | (l2p == expected_master.unsqueeze(1))).all().item())

    redundant_used = (p2l.view(args.num_ranks, num_local_physical)[:, args.num_local_master :] >= 0).sum()
    assert int(redundant_used.item()) <= args.num_ranks * args.num_redundant_experts_per_rank


def physical_loads_from_solution(global_loads, p2l, l2p, lcnts, quota, args, legacy):
    phys_loads = torch.zeros_like(p2l)
    valid = l2p >= 0
    if legacy:
        per_instance = global_loads.unsqueeze(1).float() / lcnts.clamp_min(1).unsqueeze(1).float()
        values = per_instance.masked_fill(~valid, 0).round().to(torch.int32)
    else:
        values = quota.masked_fill(~valid, 0)
    return phys_loads.scatter_add(0, l2p.clamp_min(0).flatten(), values.flatten())


def traffic_tokens(loads_per_rank, l2p, quota, args, legacy: bool):
    num_local_physical = args.num_local_master + args.num_redundant_experts_per_rank
    valid = l2p >= 0
    phys_rank = l2p.clamp_min(0) // num_local_physical
    total = int(loads_per_rank.sum().item())

    local_with_quota = torch.zeros((), dtype=torch.float32, device=loads_per_rank.device)
    local_without_quota = torch.zeros_like(local_with_quota)
    for src_rank in range(args.num_ranks):
        src_loads = loads_per_rank[src_rank].float()
        local_slot = (phys_rank == src_rank) & valid
        if legacy:
            local_without_quota += (src_loads * local_slot.any(dim=1).float() / l2p.size(1)).sum()
            local_with_quota = local_without_quota
            continue

        total_quota = quota.masked_fill(~valid, 0).sum(dim=1).clamp_min(1).float()
        local_quota = quota.masked_fill(~local_slot, 0).sum(dim=1).float()
        local_without_quota += (src_loads * local_quota / total_quota).sum()
        local_with_quota += torch.minimum(src_loads, local_quota).sum()

    return {
        "no_locality": total - int(local_without_quota.round().item()),
        "quota_locality": total - int(local_with_quota.round().item()),
        "total": total,
    }


def report_solution(mode, ratio, loads_per_rank, solution, timings, args):
    p2l, l2p, lcnts, quota, _, _ = solution
    legacy = mode == "legacy"
    global_loads = loads_per_rank.sum(dim=0, dtype=torch.int32)
    assert_legal(p2l, l2p, lcnts, args)

    phys_loads = physical_loads_from_solution(global_loads, p2l, l2p, lcnts, quota, args, legacy)
    num_local_physical = args.num_local_master + args.num_redundant_experts_per_rank
    rank_loads_before = global_loads.view(args.num_ranks, args.num_local_master).sum(dim=1)
    rank_loads_after = phys_loads.view(args.num_ranks, num_local_physical).sum(dim=1)
    replicas = lcnts - 1
    used_slots = int((p2l.view(args.num_ranks, num_local_physical)[:, args.num_local_master :] >= 0).sum().item())
    traffic = traffic_tokens(loads_per_rank, l2p, quota, args, legacy)
    replica_summary = summarize_vector(replicas)

    print(f"  [{mode}] solve e2e {timings['e2e_ms']:>8.3f} ms | kernel {timings['kernel_ms']:>8.3f} ms", flush=True)
    print(
        f"    rank max/mean      : {max_mean(rank_loads_before).item():.3f} -> "
        f"{max_mean(rank_loads_after).item():.3f}",
        flush=True,
    )
    print(
        f"    replicas min/med/avg/max: {replica_summary['min']:.0f}/"
        f"{replica_summary['median']:.0f}/{replica_summary['mean']:.2f}/{replica_summary['max']:.0f}",
        flush=True,
    )
    print(
        f"    slots used/total   : {used_slots}/{args.num_ranks * args.num_redundant_experts_per_rank}",
        flush=True,
    )
    print(
        f"    traffic tokens     : total={traffic['total']} "
        f"remote_no_locality={traffic['no_locality']} "
        f"remote_quota_locality={traffic['quota_locality']}",
        flush=True,
    )


def run_case(mode: str, ratio: float, args):
    legacy = mode == "legacy"
    loads_per_rank = generate_loads_per_rank_zipf(
        args.num_ranks,
        args.num_experts,
        args.num_local_master,
        args.topk,
        args.tokens_per_rank,
        args.variable_num_tokens,
        ratio,
        args.seed,
    )

    solution = solve_once(loads_per_rank, args, legacy)
    torch.cuda.synchronize()

    avg, _, _ = bench(
        lambda: solve_once(loads_per_rank, args, legacy),
        num_warmups=args.warmup_iters,
        num_tests=args.bench_iters,
        use_barrier=False,
    )
    if legacy:
        kernel = avg
    else:
        kernel = bench_kineto(
            lambda: solve_once(loads_per_rank, args, legacy),
            kernel_names="quota_placement_solve_kernel",
            num_tests=max(3, min(args.bench_iters, 30)),
            suppress_kineto_output=True,
        )
    report_solution(
        mode,
        ratio,
        loads_per_rank,
        solution,
        {"e2e_ms": avg * 1000, "kernel_ms": kernel * 1000},
        args,
    )


def main():
    parser = argparse.ArgumentParser(description="UltraEP placement solving test")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--num-ranks", type=int, default=64)
    parser.add_argument("--nvl-domain-size", type=int, default=64)
    parser.add_argument("--num-redundant-experts-per-rank", type=int, default=2)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tokens-per-rank", type=int, default=8192)
    parser.add_argument("--variable-num-tokens", action="store_true")
    parser.add_argument(
        "--imbalance-ratios",
        type=float,
        nargs="+",
        default=[0.0, 1.5, 2.0, 2.5, 3.0],
        help="Space-separated target rank-level max/mean ratios.",
    )
    parser.add_argument("--modes", type=str, default="quota")
    parser.add_argument("--balance-threshold", type=float, default=1.0)
    parser.add_argument("--quota-min-tokens-per-replica", type=int, default=1024)
    parser.add_argument("--quota-allow-zero-master-quota", action="store_true", default=False)
    parser.add_argument("--quota-locality-aware", action="store_true", default=True)
    parser.add_argument("--quota-oracle-eps", type=float, default=0.01)
    parser.add_argument("--quota-kernel-stage", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    if args.num_experts % args.num_ranks != 0:
        raise ValueError("--num-experts must be divisible by --num-ranks")
    args.num_local_master = args.num_experts // args.num_ranks
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    modes = parse_csv_strings(args.modes)
    for mode in modes:
        if mode not in ("quota", "legacy"):
            raise ValueError("--modes entries must be quota or legacy")
    print("=" * 96, flush=True)
    print(
        f"UltraEP Solving Test | ranks={args.num_ranks} nvl={args.nvl_domain_size} "
        f"experts={args.num_experts} local_master/rank={args.num_local_master} "
        f"redundant/rank={args.num_redundant_experts_per_rank} topk={args.topk}",
        flush=True,
    )
    print("=" * 96, flush=True)
    for ratio in args.imbalance_ratios:
        print("", flush=True)
        print("-" * 96, flush=True)
        print(f"Imbalance Ratio {ratio:g}", flush=True)
        print("-" * 96, flush=True)
        for mode in modes:
            run_case(mode, ratio, args)


if __name__ == "__main__":
    main()
