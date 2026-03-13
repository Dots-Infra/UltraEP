"""
Unit tests for the PlacementSolver (EPLB-style expert placement algorithm).

Tests correctness invariants, load-balance quality, and CPU latency across
different expert load distributions.

Pass --gpu to also test the GPU-based PlacementSolverGPU.
"""

import argparse
import time
import sys
from typing import Dict
from dataclasses import dataclass

import torch
import numpy as np

try:
    import ultra_ep._C as _C
except ImportError:
    print(
        "ERROR: Cannot import ultra_ep._C. "
        "Make sure UltraEP is built and installed (pip install -e .).",
        file=sys.stderr,
    )
    sys.exit(1)


@dataclass
class EPConfig:
    num_ranks: int
    num_local_master: int
    num_local_redundant: int
    num_nvl_ranks: int


# ============================================================================
# Helper: create solver + output tensors for a given configuration
# ============================================================================


def make_solver_and_buffers(
    config: EPConfig,
):
    num_global_logical = config.num_local_master * config.num_ranks
    num_local_physical = config.num_local_master + config.num_local_redundant
    num_global_physical = num_local_physical * config.num_ranks
    max_replicas_dim = config.num_ranks

    solver = _C.PlacementSolver(
        num_global_logical,
        config.num_ranks,
        config.num_local_master,
        config.num_local_redundant,
        config.num_nvl_ranks,
        max_replicas_dim,
    )

    p2l = torch.full((num_global_physical,), -1, dtype=torch.int32)
    l2p = torch.full((num_global_logical, max_replicas_dim), -1, dtype=torch.int32)
    lcnts = torch.zeros(num_global_logical, dtype=torch.int32)

    return solver, p2l, l2p, lcnts


def run_solver(
    solver,
    expert_loads: torch.Tensor,
    p2l: torch.Tensor,
    l2p: torch.Tensor,
    lcnts: torch.Tensor,
    balance_threshold: float = 1.0,
):
    """Run solver and return the output tensors (modified in-place)."""
    solver.solve(expert_loads, p2l, l2p, lcnts, balance_threshold)
    return p2l, l2p, lcnts


# ============================================================================
# Correctness validator
# ============================================================================


def validate_placement(
    expert_loads: torch.Tensor,
    p2l: torch.Tensor,
    l2p: torch.Tensor,
    lcnts: torch.Tensor,
    config: EPConfig,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Validate all placement invariants. Raises AssertionError on violation.
    Returns a dict of load-balance metrics.
    """
    num_global_logical = config.num_local_master * config.num_ranks
    num_local_physical = config.num_local_master + config.num_local_redundant
    num_global_physical = num_local_physical * config.num_ranks
    max_replicas_dim = config.num_ranks

    # ------------------------------------------------------------------
    # 1. Master positions are correct
    # ------------------------------------------------------------------
    for l in range(num_global_logical):
        rank = l // config.num_local_master
        local_idx = l % config.num_local_master
        p = rank * num_local_physical + local_idx
        assert (
            p2l[p].item() == l
        ), f"Master position wrong: p2l[{p}]={p2l[p].item()}, expected {l}"
        assert (
            l2p[l, 0].item() == p
        ), f"Master l2p wrong: l2p[{l},0]={l2p[l,0].item()}, expected {p}"

    # ------------------------------------------------------------------
    # 2. lcnts match actual physical count
    # ------------------------------------------------------------------
    actual_counts = torch.zeros(num_global_logical, dtype=torch.int32)
    for p in range(num_global_physical):
        l = p2l[p].item()
        if l >= 0:
            actual_counts[l] += 1
    assert (
        lcnts == actual_counts
    ).all(), (
        f"lcnts mismatch:\n  lcnts={lcnts.tolist()}\n  actual={actual_counts.tolist()}"
    )

    # ------------------------------------------------------------------
    # 3. lcnts[l] <= num_nvl_ranks for all l
    #    (each expert can appear at most once per rank in the NVL domain)
    # ------------------------------------------------------------------
    assert (lcnts <= config.num_nvl_ranks).all(), (
        f"Expert replica count exceeds num_nvl_ranks={config.num_nvl_ranks}: "
        f"max={lcnts.max().item()}"
    )

    # ------------------------------------------------------------------
    # 4. No duplicate logical expert on the same rank
    # ------------------------------------------------------------------
    for r in range(config.num_ranks):
        experts_on_rank = set()
        for local_slot in range(num_local_physical):
            p = r * num_local_physical + local_slot
            l = p2l[p].item()
            if l >= 0:
                assert l not in experts_on_rank, f"Duplicate expert {l} on rank {r}"
                experts_on_rank.add(l)

    # ------------------------------------------------------------------
    # 5. p2l and l2p are consistent
    # ------------------------------------------------------------------
    for l in range(num_global_logical):
        cnt = lcnts[l].item()
        l2p_entries = [l2p[l, k].item() for k in range(cnt)]
        for p in l2p_entries:
            assert p >= 0, f"l2p entry for expert {l} is -1 but lcnts={cnt}"
            assert (
                p2l[p].item() == l
            ), f"Inconsistency: l2p[{l}] contains {p} but p2l[{p}]={p2l[p].item()}"
        # Remaining entries should be -1
        for k in range(cnt, max_replicas_dim):
            assert (
                l2p[l, k].item() == -1
            ), f"l2p[{l},{k}]={l2p[l,k].item()} should be -1"

    # ------------------------------------------------------------------
    # 6. Replicas are not on the same rank as their master
    # ------------------------------------------------------------------
    for l in range(num_global_logical):
        master_p = l2p[l, 0].item()
        master_rank = master_p // num_local_physical
        for k in range(1, lcnts[l].item()):
            replica_p = l2p[l, k].item()
            replica_rank = replica_p // num_local_physical
            assert (
                replica_rank != master_rank
            ), f"Expert {l}: replica at phys {replica_p} is on master rank {master_rank}"

    # ------------------------------------------------------------------
    # 7. Replicas are in the same NVL domain as master
    # ------------------------------------------------------------------
    for l in range(num_global_logical):
        master_p = l2p[l, 0].item()
        master_rank = master_p // num_local_physical
        master_nvl_domain = master_rank // config.num_nvl_ranks
        for k in range(1, lcnts[l].item()):
            replica_p = l2p[l, k].item()
            replica_rank = replica_p // num_local_physical
            replica_nvl_domain = replica_rank // config.num_nvl_ranks
            assert replica_nvl_domain == master_nvl_domain, (
                f"Expert {l}: replica NVL domain {replica_nvl_domain} != "
                f"master NVL domain {master_nvl_domain}"
            )

    # ------------------------------------------------------------------
    # 8. Redundant slots filled (count per rank)
    # ------------------------------------------------------------------
    total_filled_redundant = 0
    for r in range(config.num_ranks):
        filled = 0
        for s in range(config.num_local_redundant):
            p = r * num_local_physical + config.num_local_master + s
            if p2l[p].item() >= 0:
                filled += 1
        total_filled_redundant += filled

    # ------------------------------------------------------------------
    # Compute load-balance metrics
    # ------------------------------------------------------------------
    loads_float = expert_loads.float()
    lcnts_float = lcnts.float().clamp(min=1)
    per_replica_load = loads_float / lcnts_float

    gpu_loads = torch.zeros(config.num_ranks, dtype=torch.float64)
    for p in range(num_global_physical):
        l = p2l[p].item()
        if l >= 0:
            gpu_loads[p // num_local_physical] += per_replica_load[l].item()

    max_load = gpu_loads.max().item()
    min_load = gpu_loads.min().item()
    mean_load = gpu_loads.mean().item()
    imbalance_ratio = max_load / mean_load if mean_load > 0 else 1.0

    metrics = {
        "max_gpu_load": max_load,
        "min_gpu_load": min_load,
        "mean_gpu_load": mean_load,
        "imbalance_ratio": imbalance_ratio,
        "total_filled_redundant": total_filled_redundant,
        "max_possible_redundant": config.num_ranks * config.num_local_redundant,
        "gpu_loads": gpu_loads.tolist(),
    }

    if verbose:
        print(f"  GPU loads: {[f'{x:.1f}' for x in gpu_loads.tolist()]}")
        print(f"  Imbalance ratio (max/mean): {imbalance_ratio:.4f}")
        print(
            f"  Redundant slots filled: {total_filled_redundant}/{config.num_ranks * config.num_local_redundant}"
        )

    return metrics


# ============================================================================
# Load distribution generators
# ============================================================================


def gen_uniform(num_experts: int, total_tokens: int = 10000) -> torch.Tensor:
    """Uniform load across all experts."""
    base = total_tokens // num_experts
    loads = torch.full((num_experts,), base, dtype=torch.int32)
    return loads


def gen_zipf(
    num_experts: int, alpha: float = 1.2, total_tokens: int = 100000
) -> torch.Tensor:
    """Zipf (power-law) distribution."""
    ranks = np.arange(1, num_experts + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, alpha)
    weights /= weights.sum()
    loads = (weights * total_tokens).astype(np.int32)
    loads = np.maximum(loads, 1)  # at least 1 token
    return torch.from_numpy(loads).to(torch.int32)


def gen_single_hot(
    num_experts: int,
    hot_idx: int = 0,
    hot_ratio: float = 0.8,
    total_tokens: int = 100000,
) -> torch.Tensor:
    """One expert gets hot_ratio of all tokens."""
    hot_tokens = int(total_tokens * hot_ratio)
    cold_tokens = (total_tokens - hot_tokens) // max(num_experts - 1, 1)
    loads = torch.full((num_experts,), cold_tokens, dtype=torch.int32)
    loads[hot_idx] = hot_tokens
    return loads


def gen_multi_hot(
    num_experts: int,
    num_hot: int = 4,
    hot_ratio: float = 0.6,
    total_tokens: int = 100000,
) -> torch.Tensor:
    """A few experts get the majority of tokens."""
    hot_tokens = int(total_tokens * hot_ratio) // num_hot
    cold_tokens = int(total_tokens * (1 - hot_ratio)) // max(num_experts - num_hot, 1)
    loads = torch.full((num_experts,), cold_tokens, dtype=torch.int32)
    for i in range(num_hot):
        loads[i] = hot_tokens
    return loads


def gen_alternating(
    num_experts: int, ratio: float = 10.0, total_tokens: int = 100000
) -> torch.Tensor:
    """Alternating hot/cold pattern."""
    cold = int(total_tokens / (num_experts * (1 + ratio) / 2))
    hot = int(cold * ratio)
    loads = torch.empty(num_experts, dtype=torch.int32)
    for i in range(num_experts):
        loads[i] = hot if i % 2 == 0 else cold
    return loads


def gen_random_normal(num_experts: int, total_tokens: int = 100000) -> torch.Tensor:
    """Random normal distribution (clipped to positive)."""
    mean = total_tokens / num_experts
    std = mean * 0.5
    loads = torch.normal(mean, std, size=(num_experts,)).clamp(min=1).to(torch.int32)
    return loads


DISTRIBUTIONS = {
    "uniform": gen_uniform,
    "zipf": gen_zipf,
    "single_hot": gen_single_hot,
    "multi_hot": gen_multi_hot,
    "alternating": gen_alternating,
    "random_normal": gen_random_normal,
}

# ============================================================================
# Test: correctness across distributions
# ============================================================================


def test_correctness(config: EPConfig, verbose: bool = False):
    print(f"\n{'-'*40} Correctness {'-'*40}")

    num_global_logical = config.num_local_master * config.num_ranks
    solver, p2l, l2p, lcnts = make_solver_and_buffers(config)

    all_passed = True
    for dist_name, gen_fn in DISTRIBUTIONS.items():
        loads = gen_fn(num_global_logical)
        # Reset output tensors
        p2l.fill_(-1)
        l2p.fill_(-1)
        lcnts.fill_(0)

        run_solver(solver, loads, p2l, l2p, lcnts)

        try:
            metrics = validate_placement(
                loads, p2l, l2p, lcnts, config, verbose=verbose
            )
            status = "PASS"
            detail = f"imbalance={metrics['imbalance_ratio']:.4f}"
        except AssertionError as e:
            status = "FAIL"
            detail = str(e)
            all_passed = False

        print(f"  {dist_name:20s} ... {status}  ({detail})")

    return all_passed


# ============================================================================
# Test: determinism (same input → same output)
# ============================================================================


def test_determinism(config: EPConfig):
    print(f"\n{'-'*40} Determinism {'-'*40}")

    num_global_logical = config.num_local_master * config.num_ranks
    solver, p2l_a, l2p_a, lcnts_a = make_solver_and_buffers(config)
    _, p2l_b, l2p_b, lcnts_b = make_solver_and_buffers(config)

    loads = gen_zipf(num_global_logical, alpha=1.5)
    run_solver(solver, loads, p2l_a, l2p_a, lcnts_a)
    run_solver(solver, loads, p2l_b, l2p_b, lcnts_b)

    assert (p2l_a == p2l_b).all(), "p2l not deterministic"
    assert (l2p_a == l2p_b).all(), "l2p not deterministic"
    assert (lcnts_a == lcnts_b).all(), "lcnts not deterministic"
    print("  PASS — two runs with same input produce identical output")
    return True


# ============================================================================
# Benchmark: latency
# ============================================================================


def bench_latency(config: EPConfig, num_warmup: int = 100, num_iters: int = 1000):
    print(f"\n{'-'*40} Latency {'-'*40}")

    num_global_logical = config.num_local_master * config.num_ranks
    solver, p2l, l2p, lcnts = make_solver_and_buffers(config)

    results = {}
    for dist_name, gen_fn in DISTRIBUTIONS.items():
        loads = gen_fn(num_global_logical)

        # Warmup
        for _ in range(num_warmup):
            run_solver(solver, loads, p2l, l2p, lcnts)

        # Timed runs
        times_ns = []
        for _ in range(num_iters):
            t0 = time.perf_counter_ns()
            run_solver(solver, loads, p2l, l2p, lcnts)
            t1 = time.perf_counter_ns()
            times_ns.append(t1 - t0)

        times_us = np.array(times_ns) / 1000.0
        # Remove outliers (top/bottom 5%)
        times_us = np.sort(times_us)[num_iters // 20 : -num_iters // 20]

        mean_us = np.mean(times_us)
        p50_us = np.percentile(times_us, 50)
        p99_us = np.percentile(times_us, 99)
        min_us = np.min(times_us)

        results[dist_name] = {
            "mean_us": mean_us,
            "p50_us": p50_us,
            "p99_us": p99_us,
            "min_us": min_us,
        }
        print(
            f"  {dist_name:20s}  mean={mean_us:7.2f} µs  "
            f"p50={p50_us:7.2f} µs  p99={p99_us:7.2f} µs  min={min_us:7.2f} µs"
        )

    return results


# ============================================================================
# Load balance quality report
# ============================================================================


def report_load_balance(config: EPConfig):
    print(f"\n{'-'*40} Load Balance {'-'*40}")

    num_global_logical = config.num_local_master * config.num_ranks
    solver, p2l, l2p, lcnts = make_solver_and_buffers(config)

    for dist_name, gen_fn in DISTRIBUTIONS.items():
        loads = gen_fn(num_global_logical)
        p2l.fill_(-1)
        l2p.fill_(-1)
        lcnts.fill_(0)
        run_solver(solver, loads, p2l, l2p, lcnts)

        metrics = validate_placement(loads, p2l, l2p, lcnts, config)

        # Also compute what "no replication" imbalance would be
        baseline_gpu_loads = torch.zeros(config.num_ranks, dtype=torch.float64)
        for l in range(num_global_logical):
            rank = l // config.num_local_master
            baseline_gpu_loads[rank] += loads[l].item()
        baseline_imbalance = (
            baseline_gpu_loads.max().item() / baseline_gpu_loads.mean().item()
            if baseline_gpu_loads.mean().item() > 0
            else 1.0
        )

        improvement = (
            baseline_imbalance / metrics["imbalance_ratio"]
            if metrics["imbalance_ratio"] > 0
            else 0
        )
        print(
            f"  {dist_name:20s}  "
            f"imbalance={metrics['imbalance_ratio']:.4f}  "
            f"baseline={baseline_imbalance:.4f}  "
            f"improvement={improvement:.2f}x  "
            f"redundant_fill={metrics['total_filled_redundant']}/{metrics['max_possible_redundant']}"
        )


# ============================================================================
# GPU Solver helpers
# ============================================================================


def make_gpu_solver_and_buffers(config: EPConfig):
    """Create PlacementSolverGPU + CUDA output tensors."""
    num_global_logical = config.num_local_master * config.num_ranks
    num_local_physical = config.num_local_master + config.num_local_redundant
    num_global_physical = num_local_physical * config.num_ranks
    max_replicas_dim = config.num_ranks

    solver_gpu = _C.PlacementSolverGPU(
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

    return solver_gpu, p2l, l2p, lcnts


def run_gpu_solver(
    solver_gpu,
    expert_loads: torch.Tensor,
    p2l: torch.Tensor,
    l2p: torch.Tensor,
    lcnts: torch.Tensor,
    balance_threshold: float = 1.0,
):
    """Run GPU solver and synchronize."""
    solver_gpu.solve(expert_loads, p2l, l2p, lcnts, balance_threshold)
    torch.cuda.synchronize()
    return p2l, l2p, lcnts


# ============================================================================
# GPU Solver: correctness
# ============================================================================


def test_gpu_correctness(config: EPConfig, verbose: bool = False) -> bool:
    print(f"\n{'-'*40} GPU Correctness {'-'*40}")

    if not torch.cuda.is_available():
        print("  SKIP — no CUDA device available")
        return True

    num_global_logical = config.num_local_master * config.num_ranks
    solver_gpu, p2l, l2p, lcnts = make_gpu_solver_and_buffers(config)

    all_passed = True
    for dist_name, gen_fn in DISTRIBUTIONS.items():
        loads = gen_fn(num_global_logical)
        loads_gpu = loads.cuda()

        # Reset
        p2l.fill_(-1)
        l2p.fill_(-1)
        lcnts.fill_(0)

        run_gpu_solver(solver_gpu, loads_gpu, p2l, l2p, lcnts)

        # Validate on CPU
        p2l_cpu = p2l.cpu()
        l2p_cpu = l2p.cpu()
        lcnts_cpu = lcnts.cpu()

        try:
            metrics = validate_placement(
                loads, p2l_cpu, l2p_cpu, lcnts_cpu, config, verbose=verbose
            )
            status = "PASS"
            detail = f"imbalance={metrics['imbalance_ratio']:.4f}"
        except AssertionError as e:
            status = "FAIL"
            detail = str(e)
            all_passed = False

        print(f"  {dist_name:20s} ... {status}  ({detail})")

    return all_passed


# ============================================================================
# GPU Solver: determinism
# ============================================================================


def test_gpu_determinism(config: EPConfig) -> bool:
    print(f"\n{'-'*40} GPU Determinism {'-'*40}")

    if not torch.cuda.is_available():
        print("  SKIP — no CUDA device available")
        return True

    num_global_logical = config.num_local_master * config.num_ranks
    solver_gpu, p2l_a, l2p_a, lcnts_a = make_gpu_solver_and_buffers(config)
    _, p2l_b, l2p_b, lcnts_b = make_gpu_solver_and_buffers(config)

    loads = gen_zipf(num_global_logical, alpha=1.5)
    loads_gpu = loads.cuda()

    run_gpu_solver(solver_gpu, loads_gpu, p2l_a, l2p_a, lcnts_a)
    run_gpu_solver(solver_gpu, loads_gpu, p2l_b, l2p_b, lcnts_b)

    ok = True
    if not torch.equal(p2l_a, p2l_b):
        print("  FAIL — p2l not deterministic")
        ok = False
    if not torch.equal(l2p_a, l2p_b):
        print("  FAIL — l2p not deterministic")
        ok = False
    if not torch.equal(lcnts_a, lcnts_b):
        print("  FAIL — lcnts not deterministic")
        ok = False
    if ok:
        print("  PASS — two runs with same input produce identical output")
    return ok


# ============================================================================
# GPU vs CPU quality comparison
# ============================================================================


def compare_gpu_cpu_quality(config: EPConfig) -> bool:
    """GPU solver imbalance must not be worse than 1.01× CPU solver."""
    print(f"\n{'-'*40} GPU vs CPU Quality {'-'*40}")

    if not torch.cuda.is_available():
        print("  SKIP — no CUDA device available")
        return True

    num_global_logical = config.num_local_master * config.num_ranks
    solver_cpu, p2l_cpu_t, l2p_cpu_t, lcnts_cpu_t = make_solver_and_buffers(config)
    solver_gpu, p2l_gpu_t, l2p_gpu_t, lcnts_gpu_t = make_gpu_solver_and_buffers(config)

    all_passed = True
    TOLERANCE = 1.01

    for dist_name, gen_fn in DISTRIBUTIONS.items():
        loads = gen_fn(num_global_logical)

        # CPU solver
        p2l_cpu_t.fill_(-1)
        l2p_cpu_t.fill_(-1)
        lcnts_cpu_t.fill_(0)
        run_solver(solver_cpu, loads, p2l_cpu_t, l2p_cpu_t, lcnts_cpu_t)
        cpu_metrics = validate_placement(loads, p2l_cpu_t, l2p_cpu_t, lcnts_cpu_t, config)

        # GPU solver
        p2l_gpu_t.fill_(-1)
        l2p_gpu_t.fill_(-1)
        lcnts_gpu_t.fill_(0)
        run_gpu_solver(solver_gpu, loads.cuda(), p2l_gpu_t, l2p_gpu_t, lcnts_gpu_t)
        gpu_metrics = validate_placement(
            loads, p2l_gpu_t.cpu(), l2p_gpu_t.cpu(), lcnts_gpu_t.cpu(), config
        )

        cpu_imb = cpu_metrics["imbalance_ratio"]
        gpu_imb = gpu_metrics["imbalance_ratio"]
        passed = gpu_imb <= cpu_imb * TOLERANCE
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(
            f"  {dist_name:20s} ... {status}  "
            f"cpu={cpu_imb:.4f}  gpu={gpu_imb:.4f}  "
            f"ratio={gpu_imb/max(cpu_imb,1e-9):.3f}x"
        )

    return all_passed


# ============================================================================
# GPU Solver: latency benchmark
# ============================================================================


def bench_gpu_latency(config: EPConfig, num_warmup: int = 50, num_iters: int = 200):
    print(f"\n{'-'*40} GPU Latency {'-'*40}")

    if not torch.cuda.is_available():
        print("  SKIP — no CUDA device available")
        return

    num_global_logical = config.num_local_master * config.num_ranks
    solver_gpu, p2l, l2p, lcnts = make_gpu_solver_and_buffers(config)

    results = {}
    for dist_name, gen_fn in DISTRIBUTIONS.items():
        loads = gen_fn(num_global_logical).cuda()

        # Warmup
        for _ in range(num_warmup):
            p2l.fill_(-1); l2p.fill_(-1); lcnts.fill_(0)
            solver_gpu.solve(loads, p2l, l2p, lcnts)
        torch.cuda.synchronize()

        # Timed runs using CUDA events
        times_us = []
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        for _ in range(num_iters):
            p2l.fill_(-1); l2p.fill_(-1); lcnts.fill_(0)
            start_evt.record()
            solver_gpu.solve(loads, p2l, l2p, lcnts)
            end_evt.record()
            torch.cuda.synchronize()
            times_us.append(start_evt.elapsed_time(end_evt) * 1000.0)  # ms→µs

        times_arr = np.array(times_us)
        # Remove outliers (top/bottom 5%)
        n_trim = max(1, num_iters // 20)
        times_arr = np.sort(times_arr)[n_trim:-n_trim]

        mean_us = np.mean(times_arr)
        p50_us = np.percentile(times_arr, 50)
        p99_us = np.percentile(times_arr, 99)
        min_us = np.min(times_arr)

        results[dist_name] = {"mean_us": mean_us, "p50_us": p50_us, "p99_us": p99_us, "min_us": min_us}
        print(
            f"  {dist_name:20s}  mean={mean_us:7.2f} µs  "
            f"p50={p50_us:7.2f} µs  p99={p99_us:7.2f} µs  min={min_us:7.2f} µs"
        )

    return results


# ============================================================================
# GPU vs CPU end-to-end latency comparison (solver only, no allreduce)
# ============================================================================


def bench_solver_comparison(config: EPConfig, num_warmup: int = 50, num_iters: int = 200):
    print(f"\n{'-'*40} GPU vs CPU Solver Latency Comparison {'-'*40}")

    if not torch.cuda.is_available():
        print("  SKIP — no CUDA device available")
        return

    num_global_logical = config.num_local_master * config.num_ranks
    solver_cpu, p2l_cpu_t, l2p_cpu_t, lcnts_cpu_t = make_solver_and_buffers(config)
    solver_gpu, p2l_gpu_t, l2p_gpu_t, lcnts_gpu_t = make_gpu_solver_and_buffers(config)

    loads_cpu = gen_zipf(num_global_logical, alpha=1.2)
    loads_gpu = loads_cpu.cuda()

    # CPU benchmark
    for _ in range(num_warmup):
        run_solver(solver_cpu, loads_cpu, p2l_cpu_t, l2p_cpu_t, lcnts_cpu_t)

    cpu_times_ns = []
    for _ in range(num_iters):
        t0 = time.perf_counter_ns()
        run_solver(solver_cpu, loads_cpu, p2l_cpu_t, l2p_cpu_t, lcnts_cpu_t)
        cpu_times_ns.append(time.perf_counter_ns() - t0)

    cpu_times_us = np.sort(np.array(cpu_times_ns) / 1000.0)[num_iters // 20:-num_iters // 20]

    # GPU benchmark
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(num_warmup):
        p2l_gpu_t.fill_(-1); l2p_gpu_t.fill_(-1); lcnts_gpu_t.fill_(0)
        solver_gpu.solve(loads_gpu, p2l_gpu_t, l2p_gpu_t, lcnts_gpu_t)
    torch.cuda.synchronize()

    gpu_times_us = []
    for _ in range(num_iters):
        p2l_gpu_t.fill_(-1); l2p_gpu_t.fill_(-1); lcnts_gpu_t.fill_(0)
        start_evt.record()
        solver_gpu.solve(loads_gpu, p2l_gpu_t, l2p_gpu_t, lcnts_gpu_t)
        end_evt.record()
        torch.cuda.synchronize()
        gpu_times_us.append(start_evt.elapsed_time(end_evt) * 1000.0)

    gpu_times_us = np.sort(np.array(gpu_times_us))[num_iters // 20:-num_iters // 20]

    print(f"  CPU solver:  mean={np.mean(cpu_times_us):7.2f} µs  p50={np.percentile(cpu_times_us,50):7.2f} µs  min={np.min(cpu_times_us):7.2f} µs")
    print(f"  GPU solver:  mean={np.mean(gpu_times_us):7.2f} µs  p50={np.percentile(gpu_times_us,50):7.2f} µs  min={np.min(gpu_times_us):7.2f} µs")
    speedup = np.mean(cpu_times_us) / max(np.mean(gpu_times_us), 0.001)
    print(f"  Speedup (GPU kernel alone): {speedup:.2f}x")
    print(f"  Note: end-to-end benefit is larger — GPU path also eliminates D2H/sync/H2D (~30-70µs)")


# ============================================================================
# Early-stop balance_threshold tests
# ============================================================================


def test_early_stop_balanced_load(config: EPConfig) -> bool:
    """均衡 load 下 threshold=1.1 应导致 0 replica。"""
    print(f"\n{'-'*40} Early-Stop: Balanced Load {'-'*40}")

    num_global_logical = config.num_local_master * config.num_ranks
    solver, p2l, l2p, lcnts = make_solver_and_buffers(config)

    # Perfectly uniform load
    loads = gen_uniform(num_global_logical, total_tokens=100000)
    run_solver(solver, loads, p2l, l2p, lcnts, balance_threshold=1.1)

    total_replicas = lcnts.sum().item() - num_global_logical
    try:
        metrics = validate_placement(loads, p2l, l2p, lcnts, config)
        # With perfectly uniform loads, all experts have LPR = load/1 = load,
        # and avg_per_slot = total_load / (G * M) = load.
        # So best_score = load <= load * 1.1 → early stop immediately, 0 replicas.
        assert total_replicas == 0, (
            f"Expected 0 replicas for uniform load with threshold=1.1, got {total_replicas}"
        )
        print(f"  PASS — {total_replicas} replicas (expected 0)")
        return True
    except AssertionError as e:
        print(f"  FAIL — {e}")
        return False


def test_early_stop_skewed_load(config: EPConfig) -> bool:
    """不均衡 load 下 threshold=1.2 仍分配 replica。"""
    print(f"\n{'-'*40} Early-Stop: Skewed Load {'-'*40}")

    num_global_logical = config.num_local_master * config.num_ranks
    solver, p2l, l2p, lcnts = make_solver_and_buffers(config)

    # Highly skewed load
    loads = gen_single_hot(num_global_logical, hot_idx=0, hot_ratio=0.8)
    run_solver(solver, loads, p2l, l2p, lcnts, balance_threshold=1.2)

    total_replicas = lcnts.sum().item() - num_global_logical
    try:
        metrics = validate_placement(loads, p2l, l2p, lcnts, config)
        assert total_replicas > 0, (
            f"Expected >0 replicas for skewed load with threshold=1.2, got {total_replicas}"
        )
        print(
            f"  PASS — {total_replicas} replicas allocated, "
            f"imbalance={metrics['imbalance_ratio']:.4f}"
        )
        return True
    except AssertionError as e:
        print(f"  FAIL — {e}")
        return False


def test_early_stop_disabled(config: EPConfig) -> bool:
    """threshold=1.0 (default) 与无 threshold 参数结果一致。"""
    print(f"\n{'-'*40} Early-Stop: Disabled (threshold=1.0) {'-'*40}")

    num_global_logical = config.num_local_master * config.num_ranks
    solver, p2l_a, l2p_a, lcnts_a = make_solver_and_buffers(config)
    _, p2l_b, l2p_b, lcnts_b = make_solver_and_buffers(config)

    loads = gen_zipf(num_global_logical, alpha=1.5)

    # Run with threshold=1.0 (explicit default)
    run_solver(solver, loads, p2l_a, l2p_a, lcnts_a, balance_threshold=1.0)
    # Run without threshold (default)
    solver.solve(loads, p2l_b, l2p_b, lcnts_b)

    ok = True
    if not torch.equal(p2l_a, p2l_b):
        print("  FAIL — p2l differs")
        ok = False
    if not torch.equal(l2p_a, l2p_b):
        print("  FAIL — l2p differs")
        ok = False
    if not torch.equal(lcnts_a, lcnts_b):
        print("  FAIL — lcnts differs")
        ok = False
    if ok:
        print("  PASS — threshold=1.0 produces same result as default")
    return ok


def test_early_stop_reduces_replicas(config: EPConfig) -> bool:
    """threshold > 1 产生的 replica 数 ≤ threshold=1 时的 replica 数。"""
    print(f"\n{'-'*40} Early-Stop: Reduces Replicas {'-'*40}")

    num_global_logical = config.num_local_master * config.num_ranks
    solver, p2l_full, l2p_full, lcnts_full = make_solver_and_buffers(config)
    _, p2l_es, l2p_es, lcnts_es = make_solver_and_buffers(config)

    all_passed = True
    for dist_name, gen_fn in DISTRIBUTIONS.items():
        loads = gen_fn(num_global_logical)

        # Full replication (no early-stop)
        p2l_full.fill_(-1); l2p_full.fill_(-1); lcnts_full.fill_(0)
        run_solver(solver, loads, p2l_full, l2p_full, lcnts_full, balance_threshold=1.0)
        full_replicas = lcnts_full.sum().item() - num_global_logical

        # With early-stop
        p2l_es.fill_(-1); l2p_es.fill_(-1); lcnts_es.fill_(0)
        run_solver(solver, loads, p2l_es, l2p_es, lcnts_es, balance_threshold=1.3)
        es_replicas = lcnts_es.sum().item() - num_global_logical

        # Validate correctness of early-stop result
        try:
            validate_placement(loads, p2l_es, l2p_es, lcnts_es, config)
        except AssertionError as e:
            print(f"  {dist_name:20s} ... FAIL (invariant violation: {e})")
            all_passed = False
            continue

        passed = es_replicas <= full_replicas
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(
            f"  {dist_name:20s} ... {status}  "
            f"full={full_replicas}  early_stop={es_replicas}"
        )

    return all_passed


# ============================================================================
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="PlacementSolver unit tests")
    parser.add_argument("--num-ranks", type=int, default=32, help="Number of ranks")
    parser.add_argument(
        "--num-local-master", type=int, default=4, help="Number of local master experts"
    )
    parser.add_argument(
        "--num-local-redundant",
        type=int,
        default=2,
        help="Number of local redundant experts",
    )
    parser.add_argument(
        "--nvl-domain-size", type=int, default=32, help="Number of NVL ranks"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed per-GPU loads"
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Also run GPU solver (PlacementSolverGPU) tests"
    )
    args = parser.parse_args()

    cfg = EPConfig(
        num_ranks=args.num_ranks,
        num_local_master=args.num_local_master,
        num_local_redundant=args.num_local_redundant,
        num_nvl_ranks=args.nvl_domain_size,
    )
    print(f"Config: {cfg}")

    all_passed = True

    # --- CPU solver tests (always run) ---
    if not test_correctness(cfg, verbose=args.verbose):
        all_passed = False
    test_determinism(cfg)
    report_load_balance(cfg)
    bench_latency(cfg)

    # --- Early-stop tests (always run) ---
    if not test_early_stop_balanced_load(cfg):
        all_passed = False
    if not test_early_stop_skewed_load(cfg):
        all_passed = False
    if not test_early_stop_disabled(cfg):
        all_passed = False
    if not test_early_stop_reduces_replicas(cfg):
        all_passed = False

    # --- GPU solver tests (--gpu flag) ---
    if args.gpu:
        if not test_gpu_correctness(cfg, verbose=args.verbose):
            all_passed = False
        if not test_gpu_determinism(cfg):
            all_passed = False
        if not compare_gpu_cpu_quality(cfg):
            all_passed = False
        bench_gpu_latency(cfg)
        bench_solver_comparison(cfg)

    if all_passed:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
