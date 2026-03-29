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


def create_manager(
    group,
    num_layers,
    num_local_master,
    num_local_redundant,
    use_gpu_solver=False,
    use_quota_eplb_solver=False,
):
    """Create a Manager with CUDA reroute enabled and minimal expert buffer sizes."""
    return ultra_ep.Manager(
        group=group,
        num_layers=num_layers,
        num_local_master_experts=num_local_master,
        num_local_redundant_experts=num_local_redundant,
        expert_fc1_numel=64,
        expert_fc2_numel=64,
        explicitly_destroy=True,
        use_gpu_solver=use_gpu_solver,
        use_quota_eplb_solver=use_quota_eplb_solver,
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


def routing_map_to_topk_ids(routing_map, topk):
    """Convert a fixed-topk routing map to contiguous [T, topk] logical expert IDs."""
    token_and_expert = routing_map.nonzero(as_tuple=False)
    assert token_and_expert.size(0) == routing_map.size(0) * topk
    return token_and_expert[:, 1].reshape(routing_map.size(0), topk).to(torch.int64)


def update_placement_with_random_loads(mgr, layer_id, T, topk):
    """Run update_placement with random routing map."""
    L = mgr.num_global_logical_experts
    routing_map = generate_routing_map(T, L, topk)
    mgr.update_placement(layer_id, routing_map, verify_reduced_loads=True)


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
        update_placement_with_random_loads(mgr, layer_id, T, topk)

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
    if mgr.use_quota_eplb_solver:
        # In quota mode, CPU reference path is not available.
        # Instead, verify gradient consistency via finite-difference-like check:
        # forward produces a deterministic mapping, backward must be its transpose.
        print_rank0(f"\n{'='*60}")
        print_rank0(f"Test: backward correctness (quota mode)  (T={T}, topk={topk})")
        print_rank0(f"{'='*60}")

        L = mgr.num_global_logical_experts

        torch.manual_seed(7 + dist.get_rank())
        update_placement_with_random_loads(mgr, layer_id, T, topk)
        routing_map = generate_routing_map(T, L, topk)

        # CUDA path: forward + backward
        probs = torch.randn(
            T, L, dtype=torch.float32, device="cuda", requires_grad=True
        )
        exp_probs, exp_map = mgr._reroute_cuda(layer_id, probs, routing_map)
        loss = exp_probs.sum()
        loss.backward()

        assert probs.grad is not None, "No gradient computed"

        # Verify: for each active (t, l), grad_probs[t,l] should be 1.0
        # (since loss = sum of expanded_probs, and each active prob is scattered exactly once)
        grad = probs.grad
        for t in range(min(T, 100)):  # spot-check first 100 tokens
            for l in range(L):
                if routing_map[t, l]:
                    assert (
                        abs(grad[t, l].item() - 1.0) < 1e-5
                    ), f"grad[{t},{l}]={grad[t,l].item()} expected 1.0"
                else:
                    assert (
                        grad[t, l].item() == 0.0
                    ), f"grad[{t},{l}]={grad[t,l].item()} expected 0.0 (inactive)"

        print_rank0("  PASS")
        return

    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: backward correctness  (T={T}, topk={topk})")
    print_rank0(f"{'='*60}")

    L = mgr.num_global_logical_experts

    torch.manual_seed(7 + dist.get_rank())
    update_placement_with_random_loads(mgr, layer_id, T, topk)

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


def test_quota_forward_counts(mgr, layer_id, T, topk):
    """Verify that quota-mode reroute matches per-instance quota counts."""
    assert mgr.use_quota_eplb_solver

    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: quota forward counts  (T={T}, topk={topk})")
    print_rank0(f"{'='*60}")

    L = mgr.num_global_logical_experts
    routing_map = generate_routing_map(T, L, topk)
    probs = torch.randn(T, L, dtype=torch.float32, device="cuda")
    mgr.update_placement(layer_id, routing_map, verify_reduced_loads=True)
    _, expanded_routing_map = mgr._reroute_cuda(layer_id, probs, routing_map)

    per_phys_counts = expanded_routing_map.sum(dim=0, dtype=torch.int32)
    dist.all_reduce(per_phys_counts)
    per_phys_counts = per_phys_counts.cpu()
    quota = mgr.get_quota_tensor(layer_id)
    l2p = mgr.logical_to_physical_map[layer_id]
    lcnts = mgr.logical_replica_counts[layer_id]

    for logical in range(L):
        for replica_idx in range(int(lcnts[logical].item())):
            phys = int(l2p[logical, replica_idx].item())
            expected = int(quota[logical, replica_idx].item())
            actual = int(per_phys_counts[phys].item())
            assert actual == expected, (
                f"logical={logical} replica={replica_idx} phys={phys}: "
                f"expected {expected}, got {actual}"
            )

    # Additional: verify per-rank load balance
    P_local = mgr.num_local_physical_experts
    rank_loads = per_phys_counts.view(mgr.num_ranks, P_local).sum(dim=1)
    if rank_loads.sum().item() > 0:
        max_load = rank_loads.max().item()
        mean_load = rank_loads.float().mean().item()
        imbalance = max_load / mean_load if mean_load > 0 else 1.0
        print_rank0(
            f"  Per-rank load balance: max={max_load}, mean={mean_load:.1f}, imbalance={imbalance:.3f}"
        )

    print_rank0("  PASS")


def test_sparse_quota_forward_counts(mgr, layer_id, T, topk):
    """Verify sparse quota reroute hits the same per-instance quotas."""
    assert mgr.use_quota_eplb_solver

    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: sparse quota forward counts  (T={T}, topk={topk})")
    print_rank0(f"{'='*60}")

    L = mgr.num_global_logical_experts
    routing_map = generate_routing_map(T, L, topk)
    topk_ids = routing_map_to_topk_ids(routing_map, topk).contiguous()

    mgr.update_placement_sparse(layer_id, topk_ids)
    mgr.reroute_sparse(layer_id, topk_ids)

    per_phys_counts = torch.bincount(
        topk_ids.flatten(), minlength=mgr.num_global_physical_experts
    ).to(torch.int32)
    dist.all_reduce(per_phys_counts)
    per_phys_counts = per_phys_counts.cpu()
    quota = mgr.get_quota_tensor(layer_id)
    l2p = mgr.logical_to_physical_map[layer_id]
    lcnts = mgr.logical_replica_counts[layer_id]

    for logical in range(L):
        for replica_idx in range(int(lcnts[logical].item())):
            phys = int(l2p[logical, replica_idx].item())
            expected = int(quota[logical, replica_idx].item())
            actual = int(per_phys_counts[phys].item())
            assert actual == expected, (
                f"logical={logical} replica={replica_idx} phys={phys}: "
                f"expected {expected}, got {actual}"
            )

    print_rank0("  PASS")


# ============================================================================
# Benchmark: forward and forward+backward latency
# ============================================================================


def bench_latency(mgr, layer_id, T, topk, num_warmup=50, num_iters=200):
    """Benchmark CPU vs CUDA reroute latency (forward and forward+backward)."""
    L = mgr.num_global_logical_experts
    P = mgr.num_global_physical_experts

    torch.manual_seed(42 + dist.get_rank())
    update_placement_with_random_loads(mgr, layer_id, T, topk)
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


def run_benchmarks(mgr, layer_id, T, topk, num_iters):
    """Run benchmarks across multiple (T, topk) configurations and print results."""
    print_rank0(f"\n{'='*60}")
    print_rank0("Benchmark: CPU vs CUDA reroute latency")
    print_rank0(f"{'='*60}")

    L = mgr.num_global_logical_experts
    P = mgr.num_global_physical_experts

    results = bench_latency(mgr, layer_id, T, topk, num_iters=num_iters)

    if dist.get_rank() == 0:
        print(f"\n  T={T}, L={L}, P={P}, topk={topk}:")
        # Print forward
        cpu_fwd = results["CPU_fwd"]
        cuda_fwd = results["CUDA_fwd"]
        speedup_fwd = cpu_fwd["mean"] / cuda_fwd["mean"] if cuda_fwd["mean"] > 0 else 0
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
    parser.add_argument("--T", type=int, default=8192, help="Number of tokens")
    parser.add_argument("--topk", type=int, default=8, help="Top-k experts per token")
    parser.add_argument("--bench-iters", type=int, default=200)
    parser.add_argument("--gpu-solver", action="store_true")
    parser.add_argument("--quota", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    group = setup_distributed()
    world_size = dist.get_world_size()

    print_rank0(
        f"Config: world_size={world_size}, "
        f"num_local_master={args.num_local_master}, "
        f"num_local_redundant={args.num_local_redundant}, "
        f"T={args.T}, topk={args.topk}, "
        f"solver={'GPU' if args.gpu_solver else 'CPU'}, "
        f"quota={'on' if args.quota else 'off'}"
    )

    mgr = create_manager(
        group,
        args.num_layers,
        args.num_local_master,
        args.num_local_redundant,
        use_gpu_solver=args.gpu_solver,
        use_quota_eplb_solver=args.quota,
    )

    L = mgr.num_global_logical_experts
    P = mgr.num_global_physical_experts
    print_rank0(f"L={L}, P={P}")

    layer_id = 0
    all_passed = True

    # ---- Correctness tests ----
    try:
        if args.quota:
            test_quota_forward_counts(mgr, layer_id, args.T, args.topk)
            test_sparse_quota_forward_counts(mgr, layer_id, args.T, args.topk)
        else:
            test_forward_correctness(
                mgr, layer_id, args.T, args.topk, verbose=args.verbose
            )
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

    # ---- Benchmarks ----
    if not args.quota:
        run_benchmarks(mgr, layer_id, args.T, args.topk, num_iters=args.bench_iters)

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
