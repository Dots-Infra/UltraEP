"""
Unit tests for the RerouteSolver (deterministic round-robin token dispatch).

"""

import argparse
import sys
import time

import torch
import numpy as np

try:
    import ultra_ep._C as _C
    from ultra_ep.reroute import _RerouteProbsFunction
except ImportError:
    print(
        "ERROR: Cannot import ultra_ep. "
        "Make sure UltraEP is built and installed (pip install -e .).",
        file=sys.stderr,
    )
    sys.exit(1)


def test_gpu_gradcheck():
    """torch.autograd.gradcheck on GPU tensors."""
    print(f"\n{'='*60}")
    print("Test: gpu_gradcheck")
    print(f"{'='*60}")

    if not torch.cuda.is_available():
        print("  SKIP (no CUDA)")
        return

    num_logical = 3
    num_physical = 5
    max_replicas = 3

    l2p = torch.tensor([[0, 3, -1], [1, -1, -1], [2, 4, -1]], dtype=torch.int32)
    lcnts = torch.tensor([2, 1, 2], dtype=torch.int32)

    routing_map = torch.tensor(
        [
            [True, False, True],
            [True, True, False],
            [False, True, True],
        ],
        dtype=torch.bool,
    ).cuda()

    solver = _C.RerouteSolver(num_logical, num_physical, max_replicas)
    probs = torch.randn(
        3, num_logical, dtype=torch.float64, device="cuda", requires_grad=True
    )

    def func(probs):
        token_idx, logical_idx, physical_idx = solver.solve(routing_map, l2p, lcnts)
        expanded_probs = _RerouteProbsFunction.apply(
            probs, token_idx, logical_idx, physical_idx, num_physical
        )
        return expanded_probs

    result = torch.autograd.gradcheck(func, (probs,), eps=1e-6, atol=1e-4, rtol=1e-3)
    assert result, "GPU gradcheck failed"

    print("  PASS")


def test_correctness():
    """Stress test with realistic MoE sizes."""
    print(f"\n{'='*60}")
    print("Test: correctness")
    print(f"{'='*60}")

    num_ranks = 32
    num_local_master = 4
    num_local_redundant = 2
    num_nvl_ranks = 8

    num_logical = num_local_master * num_ranks
    num_local_physical = num_local_master + num_local_redundant
    num_physical = num_local_physical * num_ranks
    max_replicas = num_ranks

    # Placement
    placement_solver = _C.PlacementSolver(
        num_logical,
        num_ranks,
        num_local_master,
        num_local_redundant,
        num_nvl_ranks,
        max_replicas,
    )

    p2l = torch.full((num_physical,), -1, dtype=torch.int32)
    l2p = torch.full((num_logical, max_replicas), -1, dtype=torch.int32)
    lcnts = torch.zeros(num_logical, dtype=torch.int32)

    loads = torch.randint(100, 10000, (num_logical,), dtype=torch.int32)
    placement_solver.solve(loads, p2l, l2p, lcnts)

    # Large routing map
    T = 4096
    topk = 2
    torch.manual_seed(99)
    routing_map = torch.zeros(T, num_logical, dtype=torch.bool, device="cuda")
    for t in range(T):
        experts = torch.randperm(num_logical)[:topk]
        routing_map[t, experts] = True

    reroute_solver = _C.RerouteSolver(num_logical, num_physical, max_replicas)

    # Test correctness
    token_idx, logical_idx, physical_idx = reroute_solver.solve(routing_map, l2p, lcnts)

    N = token_idx.size(0)
    expected_N = routing_map.sum().item()
    assert N == expected_N, f"Active count: {N} != {expected_N}"

    # All physical indices should be valid
    assert (physical_idx >= 0).all() and (physical_idx < num_physical).all()

    # For each logical expert, check the round-robin pattern
    for l_idx in range(num_logical):
        cnt = lcnts[l_idx].item()
        if cnt <= 1:
            continue
        # Get physical indices assigned to this logical expert
        mask = logical_idx == l_idx
        if not mask.any():
            continue
        phys_for_l = physical_idx[mask].tolist()
        # Check round-robin: phys_for_l[k] should be l2p[l_idx, k % cnt]
        for k, p_val in enumerate(phys_for_l):
            expected_p = l2p[l_idx, k % cnt].item()
            assert (
                p_val == expected_p
            ), f"Expert {l_idx}, token {k}: got phys {p_val}, expected {expected_p}"

    print(f"  Verified {N} active pairs across {num_logical} experts")
    print("  PASS")


def bench_latency():
    """Benchmark latency for various configurations."""
    print(f"\n{'='*60}")
    print("Benchmark: latency")
    print(f"{'='*60}")

    configs = [
        # (T, num_ranks, local_master, local_redundant, nvl_ranks, topk)
        (8192, 32, 4, 2, 32, 8),
        (8192, 32, 4, 2, 8, 8),
        (8192, 64, 2, 1, 32, 8),
        (8192, 64, 4, 2, 8, 8),
    ]

    for T, num_ranks, lm, lr, nvl, topk in configs:
        L = lm * num_ranks
        num_local_physical = lm + lr
        P = num_local_physical * num_ranks
        max_replicas = num_ranks

        # Placement
        psolver = _C.PlacementSolver(L, num_ranks, lm, lr, nvl, max_replicas)
        p2l = torch.full((P,), -1, dtype=torch.int32, device="cpu")
        l2p = torch.full((L, max_replicas), -1, dtype=torch.int32, device="cpu")
        lcnts = torch.zeros(L, dtype=torch.int32, device="cpu")

        loads = torch.randint(100, 10000, (L,), dtype=torch.int32)
        psolver.solve(loads, p2l, l2p, lcnts)

        # Routing
        torch.manual_seed(42)
        routing_map = torch.zeros(T, L, dtype=torch.bool, device="cuda")
        for t in range(T):
            experts = torch.randperm(L)[:topk]
            routing_map[t, experts] = True

        rsolver = _C.RerouteSolver(L, P, max_replicas)

        num_iters = 200
        desc = f"T={T}, L={L}, P={P}, topk={topk}"

        # Warmup
        for _ in range(20):
            rsolver.solve(routing_map, l2p, lcnts)
        torch.cuda.synchronize()

        times_ns = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            rsolver.solve(routing_map, l2p, lcnts)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            times_ns.append(t1 - t0)

        times_us = np.array(times_ns) / 1000.0
        times_us = np.sort(times_us)[10:-10]
        mean_us = np.mean(times_us)
        p50 = np.percentile(times_us, 50)
        p99 = np.percentile(times_us, 99)

        print(
            f"  CUDA: {desc:45s}  "
            f"mean={mean_us:7.1f}µs  p50={p50:7.1f}µs  p99={p99:7.1f}µs"
        )


def main():
    parser = argparse.ArgumentParser(description="RerouteSolver unit tests")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tests = [
        test_gpu_gradcheck,
        test_correctness,
        bench_latency,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"  FAIL: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    main()
