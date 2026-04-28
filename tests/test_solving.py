import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
if "" in sys.path and os.path.abspath(os.getcwd()) == REPO_ROOT:
    sys.path.remove("")
import ultra_ep._C as _C

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    bench,
    bench_kineto,
    format_load_imbalance,
    generate_routing_map_zipf,
    generate_loads_per_rank_zipf,
    load_imbalance_summary,
    max_mean,
    print_section,
    rank_token_count,
    summarize_vector,
)


def solve_once(
    loads_per_rank,
    args,
    legacy: bool,
    rank_quota_source_rank: int = -1,
    locality_aware: bool | None = None,
):
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
        args.quota_locality_aware if locality_aware is None else locality_aware,
        args.quota_oracle_eps,
        args.quota_kernel_stage,
        rank_quota_source_rank,
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


def collect_rank_quota_prefixes(loads_per_rank, args, legacy: bool, locality_aware: bool):
    if legacy:
        return None
    prefixes = []
    for source_rank in range(args.num_ranks):
        prefixes.append(
            solve_once(
                loads_per_rank,
                args,
                legacy,
                rank_quota_source_rank=source_rank,
                locality_aware=locality_aware,
            )[5]
        )
    return torch.stack(prefixes, dim=0)


def allocations_from_rank_quota_prefixes(rank_quota_prefixes):
    alloc = rank_quota_prefixes.clone()
    alloc[:, :, 1:] -= rank_quota_prefixes[:, :, :-1]
    return alloc


def physical_loads_from_rank_quotas(loads_per_rank, l2p, lcnts, rank_quota_prefixes, args):
    num_local_physical = args.num_local_master + args.num_redundant_experts_per_rank
    valid = l2p >= 0
    alloc = allocations_from_rank_quota_prefixes(rank_quota_prefixes)
    alloc = alloc.masked_fill(~valid.unsqueeze(0), 0)

    last_replica = (lcnts - 1).clamp_min(0).view(1, -1, 1).expand(args.num_ranks, -1, -1)
    final_prefix = rank_quota_prefixes.gather(2, last_replica).squeeze(-1)
    assert torch.equal(final_prefix, loads_per_rank), "rank quota prefixes do not match source loads"

    phys_loads = torch.zeros(
        args.num_ranks * num_local_physical, dtype=torch.int32, device=l2p.device
    )
    phys = l2p.clamp_min(0).unsqueeze(0).expand_as(alloc)
    return phys_loads.scatter_add(0, phys.flatten(), alloc.flatten())


def remote_tokens_from_rank_quotas(rank_quota_prefixes, l2p, args):
    num_local_physical = args.num_local_master + args.num_redundant_experts_per_rank
    valid = l2p >= 0
    alloc = allocations_from_rank_quota_prefixes(rank_quota_prefixes)
    alloc = alloc.masked_fill(~valid.unsqueeze(0), 0)
    phys_rank = l2p.clamp_min(0) // num_local_physical
    source_rank = torch.arange(args.num_ranks, device=l2p.device).view(-1, 1, 1)
    remote = (phys_rank.unsqueeze(0) != source_rank) & valid.unsqueeze(0)
    return int(alloc.masked_fill(~remote, 0).sum().item())


def traffic_tokens(
    loads_per_rank,
    l2p,
    quota,
    args,
    legacy: bool,
    rank_quota_prefixes=None,
    no_locality_rank_quota_prefixes=None,
):
    num_local_physical = args.num_local_master + args.num_redundant_experts_per_rank
    valid = l2p >= 0
    phys_rank = l2p.clamp_min(0) // num_local_physical
    routed_total = int(loads_per_rank.sum().item())

    if not legacy:
        return {
            "routed_total": routed_total,
            "without_locality": remote_tokens_from_rank_quotas(
                no_locality_rank_quota_prefixes, l2p, args
            ),
            "with_locality": remote_tokens_from_rank_quotas(rank_quota_prefixes, l2p, args),
        }

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
        "routed_total": routed_total,
        "without_locality": routed_total - int(local_without_quota.round().item()),
        "with_locality": routed_total - int(local_with_quota.round().item()),
    }


def format_pct(value: int, total: int) -> str:
    return f"{(100.0 * value / total) if total > 0 else 0.0:.2f}%"


def report_solution(
    mode,
    ratio,
    loads_per_rank,
    solution,
    timings,
    args,
    rank_quota_prefixes=None,
    no_locality_rank_quota_prefixes=None,
):
    p2l, l2p, lcnts, quota, _, _ = solution
    legacy = mode == "legacy"
    global_loads = loads_per_rank.sum(dim=0, dtype=torch.int32)
    assert_legal(p2l, l2p, lcnts, args)

    if legacy:
        phys_loads = physical_loads_from_solution(global_loads, p2l, l2p, lcnts, quota, args, legacy)
    else:
        phys_loads = physical_loads_from_rank_quotas(
            loads_per_rank, l2p, lcnts, rank_quota_prefixes, args
        )
    num_local_physical = args.num_local_master + args.num_redundant_experts_per_rank
    rank_loads_before = global_loads.view(args.num_ranks, args.num_local_master).sum(dim=1)
    rank_loads_after = phys_loads.view(args.num_ranks, num_local_physical).sum(dim=1)
    replicas = lcnts - 1
    used_slots = int((p2l.view(args.num_ranks, num_local_physical)[:, args.num_local_master :] >= 0).sum().item())
    traffic = traffic_tokens(
        loads_per_rank,
        l2p,
        quota,
        args,
        legacy,
        rank_quota_prefixes,
        no_locality_rank_quota_prefixes,
    )
    replica_summary = summarize_vector(replicas)

    W = 26  # label column width (right-aligned colon)
    print(
        f"  [{mode}] "
        f"solve e2e {timings['e2e_ms']:>8.3f} ms | "
        f"placement kernel {timings['quota_kernel_ms']:>8.3f} ms | "
        f"reroute kernel {timings['reroute_kernel_ms']:>8.3f} ms",
        flush=True,
    )
    print(f"    {'rank max/mean':<{W}}: {max_mean(rank_loads_before).item():.3f} -> "
          f"{max_mean(rank_loads_after).item():.3f}", flush=True)
    print(f"    {'replicas min/med/avg/max':<{W}}: {replica_summary['min']:.0f}/"
          f"{replica_summary['median']:.0f}/{replica_summary['mean']:.2f}/{replica_summary['max']:.0f}", flush=True)
    print(f"    {'slots used/total':<{W}}: {used_slots}/{args.num_ranks * args.num_redundant_experts_per_rank}", flush=True)
    print(
        f"    {'traffic tokens':<{W}}: routed_total={traffic['routed_total']} "
        f"remote_w/o_locality={traffic['without_locality']} "
        f"({format_pct(traffic['without_locality'], traffic['routed_total'])}) "
        f"remote_w/locality={traffic['with_locality']} "
        f"({format_pct(traffic['with_locality'], traffic['routed_total'])})",
        flush=True,
    )


def bench_reroute_kernel(solution, rank_quota_prefixes, args, legacy: bool):
    _, l2p, lcnts, _, _, _ = solution
    num_local_physical = args.num_local_master + args.num_redundant_experts_per_rank
    num_global_physical = args.num_ranks * num_local_physical

    cases = []
    for source_rank in range(args.num_ranks):
        ntokens = rank_token_count(source_rank, args.tokens_per_rank, args.variable_input_tokens, args.seed)
        routing_map = generate_routing_map_zipf(
            ntokens,
            args.num_experts,
            args.num_ranks,
            args.num_local_master,
            args.topk,
            args.imbalance_ratio,
            args.seed,
            rank=source_rank,
        )
        probs = routing_map.float()
        rank_quota_prefix = solution[5] if legacy else rank_quota_prefixes[source_rank]
        cases.append((routing_map, probs, rank_quota_prefix))

    for routing_map, probs, rank_quota_prefix in cases:
        _, expanded_routing = _C.dense_reroute_for_test(
            routing_map,
            probs,
            l2p,
            lcnts,
            rank_quota_prefix,
            num_global_physical,
            not legacy,
            args.quota_reroute_interleave,
        )
        if routing_map.size(0) > 0:
            assert bool((expanded_routing.sum(dim=1) == args.topk).all().item())
    torch.cuda.synchronize()

    case_idx = 0

    def reroute_once():
        nonlocal case_idx
        routing_map, probs, rank_quota_prefix = cases[case_idx]
        case_idx = (case_idx + 1) % len(cases)
        _C.dense_reroute_for_test(
            routing_map,
            probs,
            l2p,
            lcnts,
            rank_quota_prefix,
            num_global_physical,
            not legacy,
            args.quota_reroute_interleave,
        )

    scatter_name = "dense_rr_reroute_scatter_kernel" if legacy else "dense_quota_reroute_scatter_kernel"
    parts = bench_kineto(
        reroute_once,
        ("reroute_forward_count_kernel", scatter_name),
        num_tests=max(args.num_ranks, 3, min(args.bench_iters, 30)),
        suppress_kineto_output=True,
    )
    return sum(parts)


def run_case(mode: str, ratio: float, args):
    legacy = mode == "legacy"
    args.imbalance_ratio = ratio
    token_counts = [
        rank_token_count(rank, args.tokens_per_rank, args.variable_input_tokens, args.seed)
        for rank in range(args.num_ranks)
    ]
    loads_per_rank = generate_loads_per_rank_zipf(
        args.num_ranks,
        args.num_experts,
        args.num_local_master,
        args.topk,
        args.tokens_per_rank,
        args.variable_input_tokens,
        ratio,
        args.seed,
    )
    load_summary = load_imbalance_summary(loads_per_rank, args.num_ranks, args.num_local_master)
    print_section(
        f"Imbalance Ratio {ratio:g} | tokens/rank min/mean/max = "
        f"{min(token_counts)}/{sum(token_counts) / len(token_counts):.1f}/{max(token_counts)}\n"
        f"{format_load_imbalance(load_summary)}",
    )

    solution = solve_once(loads_per_rank, args, legacy, rank_quota_source_rank=0)
    torch.cuda.synchronize()

    rank_quota_prefixes = collect_rank_quota_prefixes(
        loads_per_rank, args, legacy, args.quota_locality_aware
    )
    no_locality_rank_quota_prefixes = collect_rank_quota_prefixes(
        loads_per_rank, args, legacy, False
    )
    reroute_kernel = bench_reroute_kernel(solution, rank_quota_prefixes, args, legacy)

    avg, _, _ = bench(
        lambda: solve_once(loads_per_rank, args, legacy, rank_quota_source_rank=0),
        num_warmups=args.warmup_iters,
        num_tests=args.bench_iters,
        use_barrier=False,
    )
    if legacy:
        kernel = avg
    else:
        kernel = bench_kineto(
            lambda: solve_once(loads_per_rank, args, legacy, rank_quota_source_rank=0),
            kernel_names="quota_placement_solve_kernel",
            num_tests=max(3, min(args.bench_iters, 30)),
            suppress_kineto_output=True,
        )
    report_solution(
        mode,
        ratio,
        loads_per_rank,
        solution,
        {
            "e2e_ms": avg * 1000,
            "quota_kernel_ms": kernel * 1000,
            "reroute_kernel_ms": reroute_kernel * 1000,
        },
        args,
        rank_quota_prefixes,
        no_locality_rank_quota_prefixes,
    )


def main():
    parser = argparse.ArgumentParser(description="UltraEP placement solving test")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--num-ranks", type=int, default=64)
    parser.add_argument("--nvl-domain-size", type=int, default=64)
    parser.add_argument("--num-redundant-experts-per-rank", type=int, default=2)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tokens-per-rank", type=int, default=8192)
    parser.add_argument("--variable-input-tokens", action="store_true", dest="variable_input_tokens")
    parser.add_argument(
        "--imbalance-ratios",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0],
        help="Space-separated target rank-level max/mean ratios (must be >= 1).",
    )
    parser.add_argument("--modes", type=str, nargs="+", default=["quota"], choices=["quota", "legacy"], help="solving modes: quota or legacy")
    parser.add_argument("--balance-threshold", type=float, default=1.0)
    parser.add_argument("--quota-min-tokens-per-replica", type=int, default=1024)
    parser.add_argument("--quota-allow-zero-master-quota", action="store_true", default=False)
    parser.add_argument("--quota-locality-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quota-oracle-eps", type=float, default=0.01)
    parser.add_argument("--quota-kernel-stage", type=int, default=1)
    parser.add_argument("--quota-reroute-interleave", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()
    if any(ratio < 1.0 for ratio in args.imbalance_ratios):
        raise ValueError("--imbalance-ratios entries must be >= 1")

    if args.num_experts % args.num_ranks != 0:
        raise ValueError("--num-experts must be divisible by --num-ranks")
    args.num_local_master = args.num_experts // args.num_ranks
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    print_section(
        f"UltraEP Solving Test | ranks={args.num_ranks} nvl={args.nvl_domain_size} "
        f"experts={args.num_experts} local_master/rank={args.num_local_master} "
        f"redundant/rank={args.num_redundant_experts_per_rank} topk={args.topk}",
    )
    for ratio in args.imbalance_ratios: 
        for mode in args.modes:
            run_case(mode, ratio, args)


if __name__ == "__main__":
    main()
