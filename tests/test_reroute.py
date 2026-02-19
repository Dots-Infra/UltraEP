"""
Reroute integration tests: correctness and performance comparison of CPU vs CUDA paths.

Example:
    torchrun --nproc_per_node=4 tests/test_reroute.py \
        --num-local-master 4 --num-local-redundant 2 --T 8192 --topk 8
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

try:
    import ultra_ep
except ImportError:
    print(
        "ERROR: Cannot import ultra_ep.",
        file=sys.stderr,
    )
    sys.exit(1)


# ============================================================================
# Setup helpers
# ============================================================================


def setup_distributed():
    """Initialize distributed environment and return the default process group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return dist.group.WORLD


def create_manager(group, num_layers, num_local_master, num_local_redundant):
    """Create a Manager with CUDA reroute enabled and minimal expert buffer sizes."""
    return ultra_ep.Manager(
        group=group,
        num_layers=num_layers,
        num_local_master_experts=num_local_master,
        num_local_redundant_experts=num_local_redundant,
        expert_fc1_numel=64,
        expert_fc2_numel=64,
        explicitly_destroy=True,
    )


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


# ============================================================================
# Data generators
# ============================================================================


def generate_routing_map(T, L, topk, device="cuda"):
    """Generate a random routing map [T, L] bool with exactly topk True per row."""
    routing_map = torch.zeros(T, L, dtype=torch.bool, device=device)
    for t in range(T):
        experts = torch.randperm(L, device="cpu")[:topk]
        routing_map[t, experts] = True
    return routing_map


def update_placement_with_random_loads(mgr, layer_id):
    """Run update_placement with random loads (all-reduced across ranks)."""
    L = mgr.num_global_logical_experts
    loads = torch.randint(100, 10000, (L,), dtype=torch.int32, device="cuda")
    dist.all_reduce(loads, group=mgr.group)
    mgr.update_placement(layer_id, loads)


# ============================================================================
# Test: forward correctness (CPU == CUDA)
# ============================================================================


def test_forward_correctness(mgr, layer_id, T, topk, verbose=False):
    """Verify that CPU and CUDA forward paths produce identical outputs."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: forward correctness  (T={T}, topk={topk})")
    print_rank0(f"{'='*60}")

    L = mgr.num_global_logical_experts

    # Run multiple trials with different placements
    num_trials = 3
    for trial in range(num_trials):
        torch.manual_seed(trial * 1000 + dist.get_rank())
        update_placement_with_random_loads(mgr, layer_id)

        routing_map = generate_routing_map(T, L, topk)
        probs = torch.randn(T, L, dtype=torch.float32, device="cuda")

        exp_probs_cpu, exp_map_cpu = mgr._reroute_cpu(layer_id, probs, routing_map)
        exp_probs_cuda, exp_map_cuda = mgr._reroute_cuda(layer_id, probs, routing_map)

        # expanded_routing_map must be bit-identical
        map_match = torch.equal(exp_map_cpu, exp_map_cuda)
        assert map_match, (
            f"Trial {trial}: routing_map mismatch — "
            f"{(exp_map_cpu != exp_map_cuda).sum().item()} differing entries"
        )

        # expanded_probs must match exactly (same float scatter, no reduction)
        probs_match = torch.equal(exp_probs_cpu, exp_probs_cuda)
        if not probs_match:
            max_diff = (exp_probs_cpu - exp_probs_cuda).abs().max().item()
            assert False, f"Trial {trial}: probs mismatch — max diff = {max_diff}"

        active = exp_map_cpu.sum().item()
        if verbose:
            print_rank0(f"  trial {trial}: {active} active pairs — match ✓")

    print_rank0(f"  {num_trials} trials PASS")


# ============================================================================
# Test: backward / gradient correctness (CPU == CUDA)
# ============================================================================


def test_backward_correctness(mgr, layer_id, T, topk):
    """Verify that CPU and CUDA backward paths produce identical grad_probs."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: backward correctness  (T={T}, topk={topk})")
    print_rank0(f"{'='*60}")

    L = mgr.num_global_logical_experts

    torch.manual_seed(7 + dist.get_rank())
    update_placement_with_random_loads(mgr, layer_id)

    routing_map = generate_routing_map(T, L, topk)

    # CPU path: forward + backward
    probs_cpu = torch.randn(
        T, L, dtype=torch.float32, device="cuda", requires_grad=True
    )
    exp_probs_cpu, _ = mgr._reroute_cpu(layer_id, probs_cpu, routing_map)
    loss_cpu = exp_probs_cpu.sum()
    loss_cpu.backward()

    # CUDA path: forward + backward (with identical input)
    probs_cuda = probs_cpu.detach().clone().requires_grad_(True)
    exp_probs_cuda, _ = mgr._reroute_cuda(layer_id, probs_cuda, routing_map)
    loss_cuda = exp_probs_cuda.sum()
    loss_cuda.backward()

    assert probs_cpu.grad is not None and probs_cuda.grad is not None

    # Forward outputs should match
    assert torch.equal(
        exp_probs_cpu, exp_probs_cuda
    ), "forward probs mismatch in grad test"

    # Gradients should match exactly
    grad_match = torch.equal(probs_cpu.grad, probs_cuda.grad)
    if not grad_match:
        max_diff = (probs_cpu.grad - probs_cuda.grad).abs().max().item()
        num_diff = (probs_cpu.grad != probs_cuda.grad).sum().item()
        assert (
            False
        ), f"gradient mismatch: {num_diff} entries differ, max diff = {max_diff}"

    print_rank0("  PASS")


# ============================================================================
# Test: edge cases
# ============================================================================


def test_edge_cases(mgr, layer_id):
    """Test edge cases: empty routing, single token, etc."""
    print_rank0(f"\n{'='*60}")
    print_rank0("Test: edge cases")
    print_rank0(f"{'='*60}")

    L = mgr.num_global_logical_experts
    update_placement_with_random_loads(mgr, layer_id)

    # Case 1: all-zero routing map (no tokens routed)
    routing_map = torch.zeros(16, L, dtype=torch.bool, device="cuda")
    probs = torch.randn(16, L, dtype=torch.float32, device="cuda")
    exp_probs_cpu, exp_map_cpu = mgr._reroute_cpu(layer_id, probs, routing_map)
    exp_probs_cuda, exp_map_cuda = mgr._reroute_cuda(layer_id, probs, routing_map)
    assert exp_map_cpu.sum() == 0, "CPU: expected empty routing map"
    assert exp_map_cuda.sum() == 0, "CUDA: expected empty routing map"
    assert exp_probs_cpu.sum() == 0, "CPU: expected zero probs"
    assert exp_probs_cuda.sum() == 0, "CUDA: expected zero probs"
    print_rank0("  Case 1 (empty routing): PASS")

    # Case 2: single token routed to one expert
    routing_map = torch.zeros(1, L, dtype=torch.bool, device="cuda")
    routing_map[0, 0] = True
    probs = torch.randn(1, L, dtype=torch.float32, device="cuda")
    exp_probs_cpu, exp_map_cpu = mgr._reroute_cpu(layer_id, probs, routing_map)
    exp_probs_cuda, exp_map_cuda = mgr._reroute_cuda(layer_id, probs, routing_map)
    assert torch.equal(exp_map_cpu, exp_map_cuda), "single token: routing map mismatch"
    assert torch.equal(exp_probs_cpu, exp_probs_cuda), "single token: probs mismatch"
    print_rank0("  Case 2 (single token): PASS")

    # Case 3: all tokens routed to every expert (dense routing)
    T_small = 32
    routing_map = torch.ones(T_small, L, dtype=torch.bool, device="cuda")
    probs = torch.randn(T_small, L, dtype=torch.float32, device="cuda")
    exp_probs_cpu, exp_map_cpu = mgr._reroute_cpu(layer_id, probs, routing_map)
    exp_probs_cuda, exp_map_cuda = mgr._reroute_cuda(layer_id, probs, routing_map)
    assert torch.equal(exp_map_cpu, exp_map_cuda), "dense routing: routing map mismatch"
    assert torch.equal(exp_probs_cpu, exp_probs_cuda), "dense routing: probs mismatch"
    print_rank0("  Case 3 (dense routing): PASS")

    # Case 4: backward with empty routing (should not crash)
    routing_map = torch.zeros(4, L, dtype=torch.bool, device="cuda")
    probs = torch.randn(4, L, dtype=torch.float32, device="cuda", requires_grad=True)
    exp_probs_cuda, _ = mgr._reroute_cuda(layer_id, probs, routing_map)
    exp_probs_cuda.sum().backward()
    assert probs.grad is not None
    assert probs.grad.sum() == 0, "expected zero gradient for empty routing"
    print_rank0("  Case 4 (backward empty): PASS")


# ============================================================================
# Benchmark: forward and forward+backward latency
# ============================================================================


def bench_latency(mgr, layer_id, T, topk, num_warmup=50, num_iters=200):
    """Benchmark CPU vs CUDA reroute latency (forward and forward+backward)."""
    L = mgr.num_global_logical_experts
    P = mgr.num_global_physical_experts

    torch.manual_seed(42 + dist.get_rank())
    update_placement_with_random_loads(mgr, layer_id)
    routing_map = generate_routing_map(T, L, topk)
    probs = torch.randn(T, L, dtype=torch.float32, device="cuda")

    results = {}

    for path_name, reroute_fn in [
        ("CPU", mgr._reroute_cpu),
        ("CUDA", mgr._reroute_cuda),
    ]:
        # ---- Forward only ----
        for _ in range(num_warmup):
            reroute_fn(layer_id, probs, routing_map)
        torch.cuda.synchronize()

        times_ns = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            reroute_fn(layer_id, probs, routing_map)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            times_ns.append(t1 - t0)

        times_us = np.array(times_ns) / 1000.0
        trimmed = np.sort(times_us)[10:-10] if len(times_us) > 20 else times_us
        results[f"{path_name}_fwd"] = {
            "mean": np.mean(trimmed),
            "p50": np.percentile(trimmed, 50),
            "p99": np.percentile(trimmed, 99),
        }

        # ---- Forward + Backward ----
        for _ in range(num_warmup):
            p = probs.detach().clone().requires_grad_(True)
            out, _ = reroute_fn(layer_id, p, routing_map)
            out.sum().backward()
        torch.cuda.synchronize()

        times_ns = []
        for _ in range(num_iters):
            p = probs.detach().clone().requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            out, _ = reroute_fn(layer_id, p, routing_map)
            out.sum().backward()
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            times_ns.append(t1 - t0)

        times_us = np.array(times_ns) / 1000.0
        trimmed = np.sort(times_us)[10:-10] if len(times_us) > 20 else times_us
        results[f"{path_name}_fwd+bwd"] = {
            "mean": np.mean(trimmed),
            "p50": np.percentile(trimmed, 50),
            "p99": np.percentile(trimmed, 99),
        }

    return results


def run_benchmarks(mgr, layer_id, configs, num_iters):
    """Run benchmarks across multiple (T, topk) configurations and print results."""
    print_rank0(f"\n{'='*60}")
    print_rank0("Benchmark: CPU vs CUDA reroute latency")
    print_rank0(f"{'='*60}")

    L = mgr.num_global_logical_experts
    P = mgr.num_global_physical_experts

    for T, topk in configs:
        results = bench_latency(mgr, layer_id, T, topk, num_iters=num_iters)

        if dist.get_rank() == 0:
            print(f"\n  T={T}, L={L}, P={P}, topk={topk}:")
            # Print forward
            cpu_fwd = results["CPU_fwd"]
            cuda_fwd = results["CUDA_fwd"]
            speedup_fwd = (
                cpu_fwd["mean"] / cuda_fwd["mean"] if cuda_fwd["mean"] > 0 else 0
            )
            print(
                f"    {'CPU  fwd':20s}  mean={cpu_fwd['mean']:8.1f}µs  "
                f"p50={cpu_fwd['p50']:8.1f}µs  p99={cpu_fwd['p99']:8.1f}µs"
            )
            print(
                f"    {'CUDA fwd':20s}  mean={cuda_fwd['mean']:8.1f}µs  "
                f"p50={cuda_fwd['p50']:8.1f}µs  p99={cuda_fwd['p99']:8.1f}µs  "
                f"speedup={speedup_fwd:.2f}x"
            )
            # Print fwd+bwd
            cpu_fb = results["CPU_fwd+bwd"]
            cuda_fb = results["CUDA_fwd+bwd"]
            speedup_fb = cpu_fb["mean"] / cuda_fb["mean"] if cuda_fb["mean"] > 0 else 0
            print(
                f"    {'CPU  fwd+bwd':20s}  mean={cpu_fb['mean']:8.1f}µs  "
                f"p50={cpu_fb['p50']:8.1f}µs  p99={cpu_fb['p99']:8.1f}µs"
            )
            print(
                f"    {'CUDA fwd+bwd':20s}  mean={cuda_fb['mean']:8.1f}µs  "
                f"p50={cuda_fb['p50']:8.1f}µs  p99={cuda_fb['p99']:8.1f}µs  "
                f"speedup={speedup_fb:.2f}x"
            )


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Reroute CPU vs CUDA correctness & performance tests"
    )
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-local-master", type=int, default=4)
    parser.add_argument("--num-local-redundant", type=int, default=2)
    parser.add_argument("--T", type=int, default=4096, help="Number of tokens")
    parser.add_argument("--topk", type=int, default=8, help="Top-k experts per token")
    parser.add_argument("--bench-iters", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    group = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0(
        f"Config: world_size={world_size}, "
        f"num_local_master={args.num_local_master}, "
        f"num_local_redundant={args.num_local_redundant}, "
        f"T={args.T}, topk={args.topk}"
    )

    mgr = create_manager(
        group, args.num_layers, args.num_local_master, args.num_local_redundant
    )

    L = mgr.num_global_logical_experts
    P = mgr.num_global_physical_experts
    print_rank0(f"L={L}, P={P}")

    layer_id = 0
    all_passed = True

    # ---- Correctness tests ----
    try:
        test_forward_correctness(mgr, layer_id, args.T, args.topk, verbose=args.verbose)
    except Exception as e:
        all_passed = False
        print_rank0(f"  FAIL: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()

    try:
        test_backward_correctness(mgr, layer_id, args.T, args.topk)
    except Exception as e:
        all_passed = False
        print_rank0(f"  FAIL: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()

    try:
        test_edge_cases(mgr, layer_id)
    except Exception as e:
        all_passed = False
        print_rank0(f"  FAIL: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()

    # ---- Benchmarks ----
    # Deduplicate benchmark configs
    bench_configs = list(
        dict.fromkeys(
            [
                (args.T, args.topk),
                (4096, 2),
                (4096, 8),
                (8192, 8),
            ]
        )
    )
    run_benchmarks(mgr, layer_id, bench_configs, num_iters=args.bench_iters)

    # ---- Summary ----
    print_rank0(f"\n{'='*60}")
    if all_passed:
        print_rank0("ALL TESTS PASSED")
    else:
        print_rank0("SOME TESTS FAILED")

    mgr.destroy()
    dist.barrier()
    dist.destroy_process_group()

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
