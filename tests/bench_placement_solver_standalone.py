"""
Standalone benchmark for PlacementSolver / PlacementSolverGPU / PlacementSolverQuota.

This script does not require torch.distributed or UltraEP Manager. It directly
instantiates the placement solvers and simulates multi-rank EP workloads on a
single process by configuring the global logical/physical expert layout through
EPConfig.

Features:
  - solver-only latency benchmark (CPU / GPU / Quota)
  - placement invariant validation
  - load-balance quality reporting
  - GPU-vs-CPU quality check
  - Quota solver correctness + latency vs GPU solver

Example:
    python3 tests/bench_placement_solver_standalone.py \
        --workloads 32:4:2:32,64:4:2:64 \
        --distributions uniform,zipf,multi_hot \
        --solver both

    # 包含 quota solver benchmark
    python3 tests/bench_placement_solver_standalone.py \
        --workloads 32:4:2:32,64:4:2:64 \
        --distributions uniform,zipf,multi_hot \
        --solver both --quota
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
from test_placement_quota import split_loads_per_rank

try:
    import ultra_ep._C as _C
except ImportError:
    print("ERROR: Cannot import ultra_ep._C.", file=sys.stderr)
    sys.exit(1)


def parse_workload(spec: str) -> EPConfig:
    parts = [p for p in re.split(r"[:x/]", spec.strip()) if p]
    if len(parts) != 4:
        raise ValueError(
            f"Invalid workload spec '{spec}'. Expected format "
            "'num_ranks:num_local_master:num_local_redundant:num_nvl_ranks'."
        )
    num_ranks, num_local_master, num_local_redundant, num_nvl_ranks = map(int, parts)
    if num_ranks <= 0 or num_local_master <= 0 or num_local_redundant < 0 or num_nvl_ranks <= 0:
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
        return [parse_workload(spec) for spec in args.workloads.split(",") if spec.strip()]
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
    num_experts = config.num_ranks * config.num_local_master
    return (
        f"num_ranks={config.num_ranks}, "
        f"num_local_master={config.num_local_master}, "
        f"num_local_redundant={config.num_local_redundant}, "
        f"num_nvl_ranks={config.num_nvl_ranks}, "
        f"num_experts={num_experts}"
    )


def baseline_imbalance(expert_loads: torch.Tensor, config: EPConfig) -> float:
    gpu_loads = torch.zeros(config.num_ranks, dtype=torch.float64)
    for l in range(expert_loads.numel()):
        rank = l // config.num_local_master
        gpu_loads[rank] += expert_loads[l].item()
    mean_load = gpu_loads.mean().item()
    return gpu_loads.max().item() / mean_load if mean_load > 0 else 1.0


def summarize_times_us(times_us: List[float]) -> Dict[str, float]:
    arr = np.array(times_us, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    trim = min(max(1, arr.size // 20), max(arr.size // 2 - 1, 0))
    if trim > 0 and arr.size > 2 * trim:
        arr = np.sort(arr)[trim:-trim]
    else:
        arr = np.sort(arr)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def format_latency(stats: Dict[str, float]) -> str:
    return (
        f"mean={stats['mean']:8.2f} us  "
        f"p50={stats['p50']:8.2f} us  "
        f"p99={stats['p99']:8.2f} us  "
        f"min={stats['min']:8.2f} us"
    )


def generate_loads(dist_name: str, num_experts: int, total_tokens: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen_fn = DISTRIBUTIONS[dist_name]
    return gen_fn(num_experts, total_tokens=total_tokens)


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
    solver_gpu, p2l_gpu, l2p_gpu, lcnts_gpu = make_gpu_solver_and_buffers(config)
    loads_gpu = loads.cuda()

    p2l_gpu.fill_(-1)
    l2p_gpu.fill_(-1)
    lcnts_gpu.fill_(0)
    run_gpu_solver(solver_gpu, loads_gpu, p2l_gpu, l2p_gpu, lcnts_gpu)
    metrics = validate_placement(
        loads,
        p2l_gpu.cpu(),
        l2p_gpu.cpu(),
        lcnts_gpu.cpu(),
        config,
        verbose=verbose,
    )

    for _ in range(warmup_iters):
        p2l_gpu.fill_(-1)
        l2p_gpu.fill_(-1)
        lcnts_gpu.fill_(0)
        solver_gpu.solve(loads_gpu, p2l_gpu, l2p_gpu, lcnts_gpu)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    times_us = []
    for _ in range(bench_iters):
        p2l_gpu.fill_(-1)
        l2p_gpu.fill_(-1)
        lcnts_gpu.fill_(0)
        start_evt.record()
        solver_gpu.solve(loads_gpu, p2l_gpu, l2p_gpu, lcnts_gpu)
        end_evt.record()
        torch.cuda.synchronize()
        times_us.append(start_evt.elapsed_time(end_evt) * 1000.0)

    return summarize_times_us(times_us), metrics


# ============================================================================
# Quota solver helpers and benchmark
# ============================================================================


def make_quota_solver_and_buffers(config: EPConfig):
    """Create PlacementSolverQuota + CUDA output tensors."""
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
    l2p = torch.full((num_global_logical, max_replicas_dim), -1, dtype=torch.int32, device="cuda")
    lcnts = torch.zeros(num_global_logical, dtype=torch.int32, device="cuda")
    quota = torch.zeros((num_global_logical, max_replicas_dim), dtype=torch.int32, device="cuda")
    quota_prefix = torch.zeros_like(quota)
    rank_quota_prefix = torch.zeros_like(quota)

    return solver, p2l, l2p, lcnts, quota, quota_prefix, rank_quota_prefix


def validate_quota_tensors(
    loads: torch.Tensor,
    expert_loads_per_rank: torch.Tensor,
    lcnts: torch.Tensor,
    quota: torch.Tensor,
    quota_prefix: torch.Tensor,
    rank_quota_prefix: torch.Tensor,
    config: EPConfig,
) -> Dict[str, object]:
    """验证 quota/quota_prefix/rank_quota_prefix 的正确性。"""
    L = loads.numel()
    R = quota.size(1)
    my_rank = 0  # standalone 模式下 runtime 未初始化，solver 使用 rank 0

    errors = []
    for l in range(L):
        C = int(lcnts[l].item())
        assert C >= 1, f"expert {l}: lcnts={C} < 1"

        # quota 各 replica 之和等于 total load
        quota_sum = int(quota[l, :C].sum().item())
        total_load = int(loads[l].item())
        if quota_sum != total_load:
            errors.append(f"expert {l}: quota sum={quota_sum} != load={total_load}")

        # quota_prefix 单调不降，最后一项等于 total load
        if C > 0 and int(quota_prefix[l, C - 1].item()) != total_load:
            errors.append(
                f"expert {l}: quota_prefix[C-1]={quota_prefix[l, C-1].item()} != load={total_load}"
            )
        for j in range(1, C):
            if int(quota_prefix[l, j].item()) < int(quota_prefix[l, j - 1].item()):
                errors.append(f"expert {l}: quota_prefix not monotonic at j={j}")

        # rank_quota_prefix[C-1] 等于 my_rank 的本地 load
        local_load = int(expert_loads_per_rank[my_rank, l].item())
        if C > 0 and int(rank_quota_prefix[l, C - 1].item()) != local_load:
            errors.append(
                f"expert {l}: rank_quota_prefix[C-1]={rank_quota_prefix[l, C-1].item()} "
                f"!= local_load={local_load}"
            )

        # 超出 C 的位置应为 0
        for j in range(C, R):
            if int(quota[l, j].item()) != 0:
                errors.append(f"expert {l}: quota[{j}]={quota[l, j].item()} should be 0")

    if errors:
        raise AssertionError("\n  ".join(errors[:5]))  # 只打印前 5 个

    return {"quota_ok": True, "num_experts_checked": L}


def compute_quota_imbalance(
    l2p: torch.Tensor,    # (num_global_logical, max_replicas_dim), int32, CPU
    lcnts: torch.Tensor,  # (num_global_logical,), int32, CPU
    quota: torch.Tensor,  # (num_global_logical, max_replicas_dim), int32, CPU
    config: EPConfig,
) -> float:
    """基于实际 quota 分配计算真实负载不均衡度。

    validate_placement() 假设 round-robin 均分（load / lcnts），但 quota solver
    为每个 replica 分配不等的 quota。本函数累加每个 rank 上所有 replica 的实际
    quota 作为真实负载，并以 max / mean 返回不均衡度。
    """
    num_local_physical = config.num_local_master + config.num_local_redundant
    rank_load: List[float] = [0.0] * config.num_ranks
    L = quota.size(0)
    for l in range(L):
        C = int(lcnts[l].item())
        for j in range(C):
            phys_idx = int(l2p[l, j].item())
            if phys_idx >= 0:
                rank = phys_idx // num_local_physical
                rank_load[rank] += int(quota[l, j].item())
    total = sum(rank_load)
    if total <= 0:
        return 1.0
    mean_load = total / config.num_ranks
    return max(rank_load) / mean_load


def run_quota_benchmark(
    config: EPConfig,
    loads: torch.Tensor,
    warmup_iters: int,
    bench_iters: int,
    verbose: bool,
    locality_aware: bool = True,
    min_tokens_per_replica: int = 1,
    solver_version: int = 1,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for PlacementSolverQuota")

    solver, p2l, l2p, lcnts, quota, quota_prefix, rank_quota_prefix = (
        make_quota_solver_and_buffers(config)
    )
    loads_gpu = loads.cuda()
    # 预计算 per-rank loads（不计入 solver 计时）
    expert_loads_per_rank = split_loads_per_rank(loads, config.num_ranks).cuda()

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
            expert_loads_per_rank,
            p2l, l2p, lcnts,
            quota, quota_prefix, rank_quota_prefix,
            1.0,                     # balance_threshold
            min_tokens_per_replica,  # min_tokens_per_replica
            True,                    # allow_zero_master_quota
            locality_aware,
            solver_version,
        )

    # Warmup
    for _ in range(warmup_iters):
        reset()
        solve_once()
    torch.cuda.synchronize()

    # 正确性验证
    reset()
    solve_once()
    torch.cuda.synchronize()
    placement_metrics = validate_placement(
        loads, p2l.cpu(), l2p.cpu(), lcnts.cpu(), config, verbose=verbose
    )
    validate_quota_tensors(
        loads,
        expert_loads_per_rank.cpu(),
        lcnts.cpu(),
        quota.cpu(),
        quota_prefix.cpu(),
        rank_quota_prefix.cpu(),
        config,
    )
    # 用实际 quota 计算真实不均衡度（替代 round-robin 假设的 imbalance_ratio）
    placement_metrics["quota_imbalance"] = compute_quota_imbalance(
        l2p.cpu(), lcnts.cpu(), quota.cpu(), config
    )

    # 计时（CUDA event）
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    times_us = []
    for _ in range(bench_iters):
        reset()
        start_evt.record()
        solve_once()
        end_evt.record()
        torch.cuda.synchronize()
        times_us.append(start_evt.elapsed_time(end_evt) * 1000.0)  # ms → µs

    return summarize_times_us(times_us), placement_metrics


def print_quota_report(
    latency: Dict[str, float],
    metrics: Dict[str, object],
    baseline: float,
    gpu_latency: Dict[str, float] = None,
    locality_aware: bool = True,
) -> None:
    # 优先使用 quota-aware 不均衡度；若尚未计算（旧调用路径）则降级到 round-robin 值
    imbalance = metrics.get("quota_imbalance", metrics["imbalance_ratio"])
    improvement = baseline / imbalance if imbalance > 0 else 0.0
    locality_tag = "locality=on" if locality_aware else "locality=off"
    speedup_str = ""
    if gpu_latency is not None:
        speedup = gpu_latency["mean"] / max(latency["mean"], 1e-9)
        speedup_str = f"  vs_gpu_solver={speedup:.2f}x"
    print(
        f"  Quota({locality_tag}) {format_latency(latency)}  "
        f"imbalance={imbalance:.4f}  "
        f"fill={metrics['total_filled_redundant']}/{metrics['max_possible_redundant']}  "
        f"vs_master_only={improvement:.2f}x{speedup_str}"
    )


def print_solver_report(
    solver_name: str,
    latency: Dict[str, float],
    metrics: Dict[str, object],
    baseline: float,
) -> None:
    improvement = (
        baseline / metrics["imbalance_ratio"]
        if metrics["imbalance_ratio"] > 0
        else 0.0
    )
    print(
        f"  {solver_name:<3} {format_latency(latency)}  "
        f"imbalance={metrics['imbalance_ratio']:.4f}  "
        f"fill={metrics['total_filled_redundant']}/{metrics['max_possible_redundant']}  "
        f"vs_master_only={improvement:.2f}x"
    )


def benchmark_one_distribution(
    config: EPConfig,
    dist_name: str,
    total_tokens: int,
    warmup_iters: int,
    bench_iters: int,
    solver_mode: str,
    quality_tolerance: float,
    seed: int,
    verbose: bool,
    quota: bool = False,
    locality_aware: bool = True,
    quota_solver_version: int = 1,
) -> bool:
    num_experts = config.num_ranks * config.num_local_master
    loads = generate_loads(dist_name, num_experts, total_tokens, seed)
    baseline = baseline_imbalance(loads, config)

    print(f"\nDistribution: {dist_name}  total_tokens={int(loads.sum().item())}  baseline={baseline:.4f}")

    cpu_latency = cpu_metrics = None
    gpu_latency = gpu_metrics = None
    all_passed = True

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
        ratio = gpu_metrics["imbalance_ratio"] / max(cpu_metrics["imbalance_ratio"], 1e-9)
        latency_speedup = cpu_latency["mean"] / max(gpu_latency["mean"], 1e-9)
        quality_ok = ratio <= quality_tolerance
        print(
            f"  Compare  gpu/cpu imbalance={ratio:.3f}x  "
            f"quality={'PASS' if quality_ok else 'FAIL'}  "
            f"cpu_to_gpu_speedup={latency_speedup:.2f}x"
        )
        if not quality_ok:
            all_passed = False

    # ---- Quota solver ----
    quota_latency = quota_metrics = None
    if quota:
        if not torch.cuda.is_available():
            print("  Quota SKIP  CUDA device is not available")
            all_passed = False
        else:
            for loc in ([True, False] if config.num_local_redundant > 0 else [True]):
                try:
                    quota_latency, quota_metrics = run_quota_benchmark(
                        config, loads, warmup_iters, bench_iters, verbose,
                        locality_aware=loc,
                        solver_version=quota_solver_version,
                    )
                    print_quota_report(
                        quota_latency, quota_metrics, baseline,
                        gpu_latency=gpu_latency,
                        locality_aware=loc,
                    )
                except AssertionError as exc:
                    print(f"  Quota(locality={'on' if loc else 'off'}) FAIL  {exc}")
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
    quota_solver_version: int = 1,
) -> bool:
    print("\n" + "=" * 100)
    print(f"Workload: {format_config(config)}")
    print("=" * 100)

    ok = True
    for idx, dist_name in enumerate(distributions):
        dist_seed = seed + idx + config.num_ranks * 17 + config.num_local_master
        ok = benchmark_one_distribution(
            config=config,
            dist_name=dist_name,
            total_tokens=total_tokens,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            solver_mode=solver_mode,
            quality_tolerance=quality_tolerance,
            seed=dist_seed,
            verbose=verbose,
            quota=quota,
            locality_aware=locality_aware,
            quota_solver_version=quota_solver_version,
        ) and ok
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
    parser.add_argument("--total-tokens", type=int, default=512*8192)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--bench-iters", type=int, default=200)
    parser.add_argument("--quality-tolerance", type=float, default=1.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--quota",
        action="store_true",
        help="同时跑 PlacementSolverQuota benchmark（locality=on 和 off 各一轮）",
    )
    parser.add_argument(
        "--quota-v2",
        action="store_true",
        help="quota solver 使用 V2 kernel（solver_version=2）",
    )
    parser.add_argument(
        "--locality-aware",
        action="store_true",
        default=True,
        help="quota solver 是否启用 locality-aware 放置（默认 True，--no-locality-aware 关闭）",
    )
    parser.add_argument(
        "--no-locality-aware",
        dest="locality_aware",
        action="store_false",
    )
    args = parser.parse_args()
    if args.quota_v2:
        args.quota = True

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
    quota_solver_version = 2 if args.quota_v2 else 1
    for config in workloads:
        all_passed = benchmark_workload(
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
            quota_solver_version=quota_solver_version,
        ) and all_passed

    print("\n" + ("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"))
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
