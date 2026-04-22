"""
Standalone benchmark for PlacementSolver / PlacementSolverGPU / PlacementSolverQuota.

The quota path is intentionally trimmed to the only supported production shape:
quota placement v1 with the fastt oracle.
"""

import argparse
import re
import sys
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from test_placement import (
    DISTRIBUTIONS,
    EPConfig,
    make_gpu_solver_and_buffers,
    make_solver_and_buffers,
    run_gpu_solver,
    run_solver,
    validate_placement,
)
from test_placement_quota import (
    make_quota_solver_and_buffers,
    split_loads_per_rank,
    validate_quota_state,
)


def parse_workload(spec: str) -> EPConfig:
    parts = [part for part in re.split(r"[:x/]", spec.strip()) if part]
    if len(parts) != 4:
        raise ValueError(
            f"Invalid workload '{spec}'. Expected "
            "'num_ranks:num_local_master:num_local_redundant:num_nvl_ranks'."
        )

    num_ranks, num_local_master, num_local_redundant, num_nvl_ranks = map(int, parts)
    if (
        num_ranks <= 0
        or num_local_master <= 0
        or num_local_redundant < 0
        or num_nvl_ranks <= 0
    ):
        raise ValueError(f"Invalid workload values in '{spec}'")
    if num_ranks % num_nvl_ranks != 0:
        raise ValueError(
            f"num_ranks must be divisible by num_nvl_ranks in workload '{spec}'"
        )

    return EPConfig(
        num_ranks=num_ranks,
        num_local_master=num_local_master,
        num_local_redundant=num_local_redundant,
        num_nvl_ranks=num_nvl_ranks,
    )


def parse_workloads(args) -> List[EPConfig]:
    if args.workloads:
        return [
            parse_workload(spec) for spec in args.workloads.split(",") if spec.strip()
        ]
    return [
        EPConfig(
            num_ranks=args.num_ranks,
            num_local_master=args.num_local_master,
            num_local_redundant=args.num_local_redundant,
            num_nvl_ranks=args.nvl_domain_size,
        )
    ]


def parse_distributions(spec: str) -> List[str]:
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in names if name not in DISTRIBUTIONS]
    if unknown:
        raise ValueError(
            f"Unknown distributions: {unknown}. Available: {sorted(DISTRIBUTIONS.keys())}"
        )
    return names


def format_config(config: EPConfig) -> str:
    return (
        f"num_ranks={config.num_ranks}, "
        f"num_local_master={config.num_local_master}, "
        f"num_local_redundant={config.num_local_redundant}, "
        f"num_nvl_ranks={config.num_nvl_ranks}, "
        f"num_experts={config.num_ranks * config.num_local_master}"
    )


def baseline_imbalance(expert_loads: torch.Tensor, config: EPConfig) -> float:
    rank_loads = torch.zeros(config.num_ranks, dtype=torch.float64)
    for expert_idx in range(expert_loads.numel()):
        rank_loads[expert_idx // config.num_local_master] += expert_loads[
            expert_idx
        ].item()
    mean_load = rank_loads.mean().item()
    return rank_loads.max().item() / mean_load if mean_load > 0 else 1.0


def summarize_times_us(times_us: List[float]) -> Dict[str, float]:
    arr = np.array(times_us, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p99": 0.0, "min": 0.0}
    trim = min(max(1, arr.size // 20), max(arr.size // 2 - 1, 0))
    arr = np.sort(arr)
    if trim > 0 and arr.size > 2 * trim:
        arr = arr[trim:-trim]
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
    }


def format_latency(stats: Dict[str, float]) -> str:
    return (
        f"mean={stats['mean']:8.2f} us  "
        f"p50={stats['p50']:8.2f} us  "
        f"p99={stats['p99']:8.2f} us  "
        f"min={stats['min']:8.2f} us"
    )


def generate_loads(
    distribution_name: str, num_experts: int, total_tokens: int, seed: int
) -> torch.Tensor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return DISTRIBUTIONS[distribution_name](num_experts, total_tokens=total_tokens)


def run_cpu_benchmark(
    config: EPConfig,
    loads: torch.Tensor,
    warmup_iters: int,
    bench_iters: int,
    verbose: bool,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    solver, p2l, l2p, lcnts = make_solver_and_buffers(config)

    p2l.fill_(-1)
    l2p.fill_(-1)
    lcnts.fill_(0)
    run_solver(solver, loads, p2l, l2p, lcnts)
    metrics = validate_placement(loads, p2l, l2p, lcnts, config, verbose=verbose)

    for _ in range(warmup_iters):
        p2l.fill_(-1)
        l2p.fill_(-1)
        lcnts.fill_(0)
        run_solver(solver, loads, p2l, l2p, lcnts)

    times_us = []
    for _ in range(bench_iters):
        p2l.fill_(-1)
        l2p.fill_(-1)
        lcnts.fill_(0)
        t0 = time.perf_counter_ns()
        run_solver(solver, loads, p2l, l2p, lcnts)
        times_us.append((time.perf_counter_ns() - t0) / 1000.0)

    return summarize_times_us(times_us), metrics


def run_gpu_benchmark(
    config: EPConfig,
    loads: torch.Tensor,
    warmup_iters: int,
    bench_iters: int,
    verbose: bool,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    solver, p2l, l2p, lcnts = make_gpu_solver_and_buffers(config)
    loads_gpu = loads.cuda()

    p2l.fill_(-1)
    l2p.fill_(-1)
    lcnts.fill_(0)
    run_gpu_solver(solver, loads_gpu, p2l, l2p, lcnts)
    metrics = validate_placement(
        loads,
        p2l.cpu(),
        l2p.cpu(),
        lcnts.cpu(),
        config,
        verbose=verbose,
    )

    for _ in range(warmup_iters):
        p2l.fill_(-1)
        l2p.fill_(-1)
        lcnts.fill_(0)
        solver.solve(loads_gpu, p2l, l2p, lcnts)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    times_us = []
    for _ in range(bench_iters):
        p2l.fill_(-1)
        l2p.fill_(-1)
        lcnts.fill_(0)
        start_evt.record()
        solver.solve(loads_gpu, p2l, l2p, lcnts)
        end_evt.record()
        torch.cuda.synchronize()
        times_us.append(start_evt.elapsed_time(end_evt) * 1000.0)

    return summarize_times_us(times_us), metrics


def compute_quota_imbalance(
    l2p: torch.Tensor,
    lcnts: torch.Tensor,
    quota: torch.Tensor,
    config: EPConfig,
) -> float:
    num_local_physical = config.num_local_master + config.num_local_redundant
    rank_loads = [0.0] * config.num_ranks
    for expert_idx in range(quota.size(0)):
        replica_count = int(lcnts[expert_idx].item())
        for replica_idx in range(replica_count):
            phys_idx = int(l2p[expert_idx, replica_idx].item())
            if phys_idx >= 0:
                rank = phys_idx // num_local_physical
                rank_loads[rank] += int(quota[expert_idx, replica_idx].item())
    total = sum(rank_loads)
    if total <= 0:
        return 1.0
    mean_load = total / config.num_ranks
    return max(rank_loads) / mean_load


def run_quota_benchmark(
    config: EPConfig,
    loads: torch.Tensor,
    warmup_iters: int,
    bench_iters: int,
    verbose: bool,
    locality_aware: bool = True,
    min_tokens_per_replica: int = 1,
    allow_zero_master_quota: bool = True,
    oracle_eps: float = 0.01,
    kernel_stage: int = 1,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for PlacementSolverQuota")

    solver, p2l, l2p, lcnts, quota, quota_prefix, rank_quota_prefix = (
        make_quota_solver_and_buffers(config)
    )
    loads_gpu = loads.cuda()
    expert_loads_per_rank = split_loads_per_rank(loads, config.num_ranks)
    expert_loads_per_rank_gpu = expert_loads_per_rank.cuda()

    def reset():
        p2l.fill_(-1)
        l2p.fill_(-1)
        lcnts.fill_(0)
        quota.fill_(0)
        quota_prefix.fill_(0)
        rank_quota_prefix.fill_(0)

    def solve_once():
        solver.solve(
            loads_gpu,
            expert_loads_per_rank_gpu,
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

    for _ in range(warmup_iters):
        reset()
        solve_once()
    torch.cuda.synchronize()

    reset()
    solve_once()
    torch.cuda.synchronize()

    p2l_cpu = p2l.cpu()
    l2p_cpu = l2p.cpu()
    lcnts_cpu = lcnts.cpu()
    quota_cpu = quota.cpu()
    quota_prefix_cpu = quota_prefix.cpu()
    rank_quota_prefix_cpu = rank_quota_prefix.cpu()

    placement_metrics = validate_placement(
        loads, p2l_cpu, l2p_cpu, lcnts_cpu, config, verbose=verbose
    )
    validate_quota_state(
        config,
        loads.cpu(),
        expert_loads_per_rank.cpu(),
        p2l_cpu,
        l2p_cpu,
        lcnts_cpu,
        quota_cpu,
        quota_prefix_cpu,
        rank_quota_prefix_cpu,
        my_rank=0,
    )
    placement_metrics["quota_imbalance"] = compute_quota_imbalance(
        l2p_cpu, lcnts_cpu, quota_cpu, config
    )

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    times_us = []
    for _ in range(bench_iters):
        reset()
        start_evt.record()
        solve_once()
        end_evt.record()
        torch.cuda.synchronize()
        times_us.append(start_evt.elapsed_time(end_evt) * 1000.0)

    return summarize_times_us(times_us), placement_metrics


def print_solver_report(
    solver_name: str,
    latency: Dict[str, float],
    metrics: Dict[str, object],
    baseline: float,
):
    improvement = (
        baseline / metrics["imbalance_ratio"] if metrics["imbalance_ratio"] > 0 else 0.0
    )
    print(
        f"  {solver_name:<5} {format_latency(latency)}  "
        f"imbalance={metrics['imbalance_ratio']:.4f}  "
        f"fill={metrics['total_filled_redundant']}/{metrics['max_possible_redundant']}  "
        f"vs_master_only={improvement:.2f}x"
    )


def print_quota_report(
    latency: Dict[str, float],
    metrics: Dict[str, object],
    baseline: float,
    locality_aware: bool,
    kernel_stage: int,
):
    imbalance = metrics["quota_imbalance"]
    improvement = baseline / imbalance if imbalance > 0 else 0.0
    print(
        f"  QUOTA(locality={'on' if locality_aware else 'off'},oracle=fastt,stage={kernel_stage}) "
        f"{format_latency(latency)}  "
        f"imbalance={imbalance:.4f}  "
        f"fill={metrics['total_filled_redundant']}/{metrics['max_possible_redundant']}  "
        f"vs_master_only={improvement:.2f}x"
    )


def benchmark_one_distribution(
    config: EPConfig,
    distribution_name: str,
    total_tokens: int,
    warmup_iters: int,
    bench_iters: int,
    solver_mode: str,
    quality_tolerance: float,
    seed: int,
    verbose: bool,
    quota: bool = False,
    locality_aware: bool = True,
    min_tokens_per_replica: int = 1,
    allow_zero_master_quota: bool = True,
    quota_oracle_eps: float = 0.01,
    quota_kernel_stage: int = 1,
) -> bool:
    num_experts = config.num_ranks * config.num_local_master
    loads = generate_loads(distribution_name, num_experts, total_tokens, seed)
    baseline = baseline_imbalance(loads, config)

    print(
        f"\nDistribution: {distribution_name}  "
        f"total_tokens={int(loads.sum().item())}  baseline={baseline:.4f}"
    )

    all_passed = True
    cpu_latency = cpu_metrics = None
    gpu_latency = gpu_metrics = None

    if solver_mode in ("cpu", "both"):
        try:
            cpu_latency, cpu_metrics = run_cpu_benchmark(
                config, loads, warmup_iters, bench_iters, verbose
            )
            print_solver_report("CPU", cpu_latency, cpu_metrics, baseline)
        except AssertionError as exc:
            print(f"  CPU FAIL  {exc}")
            all_passed = False

    if solver_mode in ("gpu", "both"):
        if not torch.cuda.is_available():
            print("  GPU SKIP  CUDA device is not available")
            all_passed = False
        else:
            try:
                gpu_latency, gpu_metrics = run_gpu_benchmark(
                    config, loads, warmup_iters, bench_iters, verbose
                )
                print_solver_report("GPU", gpu_latency, gpu_metrics, baseline)
            except AssertionError as exc:
                print(f"  GPU FAIL  {exc}")
                all_passed = False

    if cpu_metrics is not None and gpu_metrics is not None:
        ratio = gpu_metrics["imbalance_ratio"] / max(
            cpu_metrics["imbalance_ratio"], 1e-9
        )
        latency_speedup = cpu_latency["mean"] / max(gpu_latency["mean"], 1e-9)
        quality_ok = ratio <= quality_tolerance
        print(
            f"  Compare  gpu/cpu imbalance={ratio:.3f}x  "
            f"quality={'PASS' if quality_ok else 'FAIL'}  "
            f"cpu_to_gpu_speedup={latency_speedup:.2f}x"
        )
        if not quality_ok:
            all_passed = False

    if quota:
        if not torch.cuda.is_available():
            print("  QUOTA SKIP  CUDA device is not available")
            all_passed = False
        else:
            try:
                quota_latency, quota_metrics = run_quota_benchmark(
                    config,
                    loads,
                    warmup_iters,
                    bench_iters,
                    verbose,
                    locality_aware=locality_aware,
                    min_tokens_per_replica=min_tokens_per_replica,
                    allow_zero_master_quota=allow_zero_master_quota,
                    oracle_eps=quota_oracle_eps,
                    kernel_stage=quota_kernel_stage,
                )
                print_quota_report(
                    quota_latency,
                    quota_metrics,
                    baseline,
                    locality_aware,
                    quota_kernel_stage,
                )
            except AssertionError as exc:
                print(f"  QUOTA FAIL  {exc}")
                all_passed = False

    return all_passed


def benchmark_workload(
    config: EPConfig,
    distributions: Iterable[str],
    total_tokens: int,
    warmup_iters: int,
    bench_iters: int,
    solver_mode: str,
    quality_tolerance: float,
    seed: int,
    verbose: bool,
    quota: bool = False,
    locality_aware: bool = True,
    min_tokens_per_replica: int = 1,
    allow_zero_master_quota: bool = True,
    quota_oracle_eps: float = 0.01,
    quota_kernel_stage: int = 1,
) -> bool:
    print("\n" + "=" * 100)
    print(f"Workload: {format_config(config)}")
    print("=" * 100)

    ok = True
    for idx, distribution_name in enumerate(distributions):
        dist_seed = seed + idx + config.num_ranks * 17 + config.num_local_master
        ok = (
            benchmark_one_distribution(
                config=config,
                distribution_name=distribution_name,
                total_tokens=total_tokens,
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
                solver_mode=solver_mode,
                quality_tolerance=quality_tolerance,
                seed=dist_seed,
                verbose=verbose,
                quota=quota,
                locality_aware=locality_aware,
                min_tokens_per_replica=min_tokens_per_replica,
                allow_zero_master_quota=allow_zero_master_quota,
                quota_oracle_eps=quota_oracle_eps,
                quota_kernel_stage=quota_kernel_stage,
            )
            and ok
        )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone PlacementSolver benchmark without distributed environment"
    )
    parser.add_argument("--workloads", type=str, default="")
    parser.add_argument("--num-ranks", type=int, default=64)
    parser.add_argument("--num-local-master", type=int, default=2)
    parser.add_argument("--num-local-redundant", type=int, default=2)
    parser.add_argument("--nvl-domain-size", type=int, default=64)
    parser.add_argument(
        "--solver",
        choices=("cpu", "gpu", "both"),
        default="both" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--distributions",
        type=str,
        default="uniform,zipf,multi_hot,alternating",
        help=f"Comma-separated names from: {','.join(sorted(DISTRIBUTIONS.keys()))}",
    )
    parser.add_argument("--total-tokens", type=int, default=512 * 8192)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--bench-iters", type=int, default=200)
    parser.add_argument("--quality-tolerance", type=float, default=1.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--quota", action="store_true", help="Also benchmark PlacementSolverQuota"
    )
    parser.add_argument(
        "--locality-aware",
        action="store_true",
        default=True,
        help="Use locality-aware quota placement (default: on)",
    )
    parser.add_argument(
        "--no-locality-aware",
        dest="locality_aware",
        action="store_false",
    )
    parser.add_argument(
        "--quota-min-tokens-per-replica",
        type=int,
        default=1,
        help="Minimum tokens before adding a quota replica",
    )
    parser.add_argument(
        "--quota-disallow-zero-master-quota",
        action="store_true",
        help="Force each master replica to keep non-zero quota",
    )
    parser.add_argument(
        "--quota-oracle-eps",
        type=float,
        default=0.01,
        help="Fastt oracle epsilon",
    )
    parser.add_argument(
        "--quota-kernel-stage",
        type=int,
        choices=(0, 1),
        default=1,
        help="Quota solver kernel stage: 0 for baseline path, 1 for the v4a-optimized path",
    )
    args = parser.parse_args()

    try:
        workloads = parse_workloads(args)
        distributions = parse_distributions(args.distributions)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.solver in ("gpu", "both") and not torch.cuda.is_available():
        print("WARNING: CUDA is not available, falling back to CPU-only mode.")
        args.solver = "cpu"

    if args.quota and not torch.cuda.is_available():
        print("WARNING: CUDA is not available, --quota will be skipped.")
        args.quota = False

    all_passed = True
    for config in workloads:
        all_passed = (
            benchmark_workload(
                config=config,
                distributions=distributions,
                total_tokens=args.total_tokens,
                warmup_iters=args.warmup_iters,
                bench_iters=args.bench_iters,
                solver_mode=args.solver,
                quality_tolerance=args.quality_tolerance,
                seed=args.seed,
                verbose=args.verbose,
                quota=args.quota,
                locality_aware=args.locality_aware,
                min_tokens_per_replica=args.quota_min_tokens_per_replica,
                allow_zero_master_quota=not args.quota_disallow_zero_master_quota,
                quota_oracle_eps=args.quota_oracle_eps,
                quota_kernel_stage=args.quota_kernel_stage,
            )
            and all_passed
        )

    print("\n" + ("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"))
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
