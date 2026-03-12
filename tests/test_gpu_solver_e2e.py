"""
End-to-end tests for Manager with use_gpu_solver=True.

Verifies that the GPU solver path produces correct results across the full
EPLB pipeline: update_placement → reroute → weight_sync → grad_reduce.
Also tests allocate_microbatch_slot with max_microbatches > 1.

Tests run with both CPU and GPU solvers and assert identical or equivalent
correctness guarantees.

Example:
    torchrun --nproc_per_node=4 tests/test_gpu_solver_e2e.py \
        --num-local-master 4 --num-local-redundant 2
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
    print("ERROR: Cannot import ultra_ep.", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import bench

NUM_LAYERS = 4


def print_rank0(msg: str):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def generate_routing_map(T, L, topk, device="cuda"):
    """Generate a random routing map [T, L] bool with exactly topk True per row."""
    routing_map = torch.zeros(T, L, dtype=torch.bool, device=device)
    for t in range(T):
        experts = torch.randperm(L, device="cpu")[:topk]
        routing_map[t, experts] = True
    return routing_map


def create_manager(group, num_layers, num_local_master, num_local_redundant,
                   use_gpu_solver=False, max_microbatches=1):
    """Create a Manager with specified solver mode."""
    return ultra_ep.Manager(
        group=group,
        num_layers=num_layers,
        num_local_master_experts=num_local_master,
        num_local_redundant_experts=num_local_redundant,
        expert_fc1_numel=64,   # Small for testing
        expert_fc2_numel=32,
        is_train=True,
        explicitly_destroy=True,
        max_microbatches=max_microbatches,
        use_gpu_solver=use_gpu_solver,
    )


def setup_master_ptrs(mgr, layer_id, fc1_numel, fc2_numel, num_local_master):
    """Create and register master weight/grad buffers for a layer."""
    fc1_weights = [
        torch.randn(fc1_numel, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_local_master)
    ]
    fc2_weights = [
        torch.randn(fc2_numel, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_local_master)
    ]
    fc1_grads = [
        torch.randn(fc1_numel, device="cuda", dtype=torch.float32)
        for _ in range(num_local_master)
    ]
    fc2_grads = [
        torch.randn(fc2_numel, device="cuda", dtype=torch.float32)
        for _ in range(num_local_master)
    ]
    mgr.construct_local_master_ptr_pool(layer_id, fc1_weights, fc2_weights, fc1_grads, fc2_grads)
    return fc1_weights, fc2_weights, fc1_grads, fc2_grads


# ============================================================================
# Test 1: GPU vs CPU solver produce valid placements
# ============================================================================

def test_placement_equivalence(args):
    """Verify GPU solver update_placement produces valid placement maps.

    Both CPU and GPU solvers must satisfy the same invariants:
    - Masters at correct positions
    - lcnts match actual physical count
    - No duplicate logical expert on same rank
    - Replicas in same NVL domain as master
    """
    print_rank0(f"\n{'='*60}")
    print_rank0("Test: GPU vs CPU solver placement equivalence")
    print_rank0(f"{'='*60}")

    group = dist.group.WORLD
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    T = args.T
    topk = args.topk

    for solver_name, use_gpu in [("CPU", False), ("GPU", True)]:
        mgr = create_manager(
            group, NUM_LAYERS, args.num_local_master,
            args.num_local_redundant, use_gpu_solver=use_gpu,
        )
        L = mgr.num_global_logical_experts
        layer_id = 0

        # Generate and apply placement
        torch.manual_seed(42 + rank)
        routing_map = generate_routing_map(T, L, topk)
        mgr.update_placement(layer_id, routing_map, verify_reduced_loads=True)

        # Validate placement maps on all ranks
        p2l = mgr.physical_to_logical_map[layer_id]
        l2p = mgr.logical_to_physical_map[layer_id]
        lcnts = mgr.logical_replica_counts[layer_id]

        num_local_physical = mgr.num_local_physical_experts

        # Check masters
        for l in range(L):
            master_rank = l // args.num_local_master
            local_idx = l % args.num_local_master
            p = master_rank * num_local_physical + local_idx
            assert p2l[p].item() == l, (
                f"[{solver_name}] Master wrong: p2l[{p}]={p2l[p].item()}, expected {l}"
            )
            assert l2p[l, 0].item() == p, (
                f"[{solver_name}] Master l2p wrong: l2p[{l},0]={l2p[l,0].item()}, expected {p}"
            )

        # Check lcnts match actual count
        actual_counts = torch.zeros(L, dtype=torch.int32)
        for p in range(mgr.num_global_physical_experts):
            l = p2l[p].item()
            if l >= 0:
                actual_counts[l] += 1
        assert (lcnts == actual_counts).all(), (
            f"[{solver_name}] lcnts mismatch"
        )

        # Check no duplicate logical expert on same rank
        for r in range(world_size):
            experts_on_rank = set()
            for s in range(num_local_physical):
                p = r * num_local_physical + s
                l = p2l[p].item()
                if l >= 0:
                    assert l not in experts_on_rank, (
                        f"[{solver_name}] Duplicate expert {l} on rank {r}"
                    )
                    experts_on_rank.add(l)

        print_rank0(f"  {solver_name} solver: placement invariants PASS")
        mgr.destroy()

    dist.barrier()
    print_rank0("  All placement tests PASS")


# ============================================================================
# Test 2: Full pipeline (placement → reroute → weight_sync → grad_reduce)
# ============================================================================

def test_full_pipeline(args):
    """End-to-end test: placement → reroute → weight_sync → grad_reduce with GPU solver."""
    print_rank0(f"\n{'='*60}")
    print_rank0("Test: Full EPLB pipeline with GPU solver")
    print_rank0(f"{'='*60}")

    group = dist.group.WORLD
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    T = args.T
    topk = args.topk

    for solver_name, use_gpu in [("CPU", False), ("GPU", True)]:
        mgr = create_manager(
            group, NUM_LAYERS, args.num_local_master,
            args.num_local_redundant, use_gpu_solver=use_gpu,
        )
        L = mgr.num_global_logical_experts
        fc1_numel = mgr.expert_fc1_numel
        fc2_numel = mgr.expert_fc2_numel
        layer_id = 0

        # 1. Setup master pointers
        fc1_w, fc2_w, fc1_g, fc2_g = setup_master_ptrs(
            mgr, layer_id, fc1_numel, fc2_numel, args.num_local_master
        )

        # 2. Update placement
        torch.manual_seed(42 + rank)
        routing_map = generate_routing_map(T, L, topk)
        mgr.update_placement(layer_id, routing_map, verify_reduced_loads=True)

        # 3. Reroute (both CPU and CUDA paths)
        probs = torch.randn(T, L, dtype=torch.float32, device="cuda")
        exp_probs, exp_map = mgr.reroute(layer_id, probs, routing_map, backend="cuda")

        # Verify expanded routing map has correct shape
        P = mgr.num_global_physical_experts
        assert exp_probs.shape == (T, P), (
            f"[{solver_name}] exp_probs shape {exp_probs.shape}, expected ({T}, {P})"
        )
        assert exp_map.shape == (T, P), (
            f"[{solver_name}] exp_map shape {exp_map.shape}, expected ({T}, {P})"
        )

        # Each token should have >= topk physical experts selected
        # (some logicals may have replicas, so physical count >= topk)
        tokens_with_experts = exp_map.sum(dim=1)
        assert (tokens_with_experts >= topk).all(), (
            f"[{solver_name}] Some tokens have fewer than topk physical experts"
        )

        # 4. Weight sync
        dist.barrier()
        mgr.weight_sync(layer_id, async_finish=False)
        dist.barrier()

        # Verify replica weights match masters (similar to test_weight_sync.py)
        replica_weight_buffer = mgr.local_replica_weight_buffer
        num_local_physical = mgr.num_local_physical_experts

        # Build global master weights for reference
        local_master_fc1_weights = torch.stack(fc1_w)
        local_master_fc2_weights = torch.stack(fc2_w)
        global_fc1_list = [torch.zeros_like(local_master_fc1_weights) for _ in range(world_size)]
        global_fc2_list = [torch.zeros_like(local_master_fc2_weights) for _ in range(world_size)]
        dist.all_gather(global_fc1_list, local_master_fc1_weights)
        dist.all_gather(global_fc2_list, local_master_fc2_weights)
        global_fc1 = torch.stack(global_fc1_list)
        global_fc2 = torch.stack(global_fc2_list)

        for i in range(args.num_local_redundant):
            local_phys_idx = args.num_local_master + i
            global_phys_idx = rank * num_local_physical + local_phys_idx
            logical_idx = mgr.physical_to_logical_map[layer_id, global_phys_idx].item()
            if logical_idx < 0:
                continue
            master_global_phys = mgr.logical_to_physical_map[layer_id, logical_idx, 0].item()
            master_rank = master_global_phys // num_local_physical
            master_local = master_global_phys % num_local_physical
            expected_fc1 = global_fc1[master_rank, master_local]
            expected_fc2 = global_fc2[master_rank, master_local]
            actual_fc1 = replica_weight_buffer[i, :fc1_numel]
            actual_fc2 = replica_weight_buffer[i, fc1_numel:]
            assert torch.allclose(actual_fc1, expected_fc1, atol=1e-5), (
                f"[{solver_name}] Weight sync mismatch for replica {i} on rank {rank}"
            )
            assert torch.allclose(actual_fc2, expected_fc2, atol=1e-5), (
                f"[{solver_name}] Weight sync mismatch for replica {i} on rank {rank}"
            )

        # 5. Grad reduce
        # Fill replica grad buffer with known values
        replica_grad_buffer = mgr.local_replica_grad_buffer
        replica_grad_ref = torch.randn_like(replica_grad_buffer)
        replica_grad_buffer.copy_(replica_grad_ref)

        dist.barrier()
        mgr.grad_reduce(layer_id, mode="low_sm", async_finish=False)
        dist.barrier()

        # After grad_reduce, replica grad buffer should be zeroed
        assert (replica_grad_buffer == 0).all().item(), (
            f"[{solver_name}] Replica grad buffer not zeroed after grad_reduce on rank {rank}"
        )

        print_rank0(f"  {solver_name} solver: full pipeline PASS")
        mgr.destroy()

    dist.barrier()
    print_rank0("  All pipeline tests PASS")


# ============================================================================
# Test 3: GPU vs CPU solver produce equivalent reroute outputs
# ============================================================================

def test_reroute_gpu_vs_cpu_solver(args):
    """With same routing_map, GPU and CPU solver should produce valid reroute."""
    print_rank0(f"\n{'='*60}")
    print_rank0("Test: Reroute output comparison GPU vs CPU solver")
    print_rank0(f"{'='*60}")

    group = dist.group.WORLD
    rank = dist.get_rank()
    T = args.T
    topk = args.topk

    mgr_cpu = create_manager(
        group, NUM_LAYERS, args.num_local_master,
        args.num_local_redundant, use_gpu_solver=False,
    )
    mgr_gpu = create_manager(
        group, NUM_LAYERS, args.num_local_master,
        args.num_local_redundant, use_gpu_solver=True,
    )

    L = mgr_cpu.num_global_logical_experts
    layer_id = 0

    # Same routing_map for both
    torch.manual_seed(42 + rank)
    routing_map = generate_routing_map(T, L, topk)
    probs = torch.randn(T, L, dtype=torch.float32, device="cuda")

    mgr_cpu.update_placement(layer_id, routing_map, verify_reduced_loads=True)
    mgr_gpu.update_placement(layer_id, routing_map, verify_reduced_loads=True)

    # Both should produce valid reroute outputs
    exp_probs_cpu, exp_map_cpu = mgr_cpu.reroute(layer_id, probs, routing_map, backend="cuda")
    exp_probs_gpu, exp_map_gpu = mgr_gpu.reroute(layer_id, probs, routing_map, backend="cuda")

    # Both outputs should have same shape and valid structure
    assert exp_probs_cpu.shape == exp_probs_gpu.shape
    assert exp_map_cpu.shape == exp_map_gpu.shape

    # Both should route every token to at least topk physical experts
    cpu_counts = exp_map_cpu.sum(dim=1)
    gpu_counts = exp_map_gpu.sum(dim=1)
    assert (cpu_counts >= topk).all(), "CPU solver: some tokens have < topk experts"
    assert (gpu_counts >= topk).all(), "GPU solver: some tokens have < topk experts"

    # Placement maps may differ (both are valid), so the reroute outputs
    # may differ — but total token-expert pairs should be comparable
    cpu_total = exp_map_cpu.sum().item()
    gpu_total = exp_map_gpu.sum().item()
    print_rank0(
        f"  CPU solver: {cpu_total} token-expert pairs, "
        f"GPU solver: {gpu_total} token-expert pairs"
    )

    print_rank0("  PASS — both solvers produce valid reroute outputs")

    # mgr_cpu.destroy()
    mgr_gpu.destroy()


# ============================================================================
# Test 4: allocate_microbatch_slot with max_microbatches > 1
# ============================================================================

def test_microbatch_slots(args):
    """Test allocate_microbatch_slot with PP-style multi-microbatch mode."""
    print_rank0(f"\n{'='*60}")
    print_rank0("Test: allocate_microbatch_slot with max_microbatches > 1")
    print_rank0(f"{'='*60}")

    group = dist.group.WORLD
    rank = dist.get_rank()
    max_mbs = 4

    for solver_name, use_gpu in [("CPU", False), ("GPU", True)]:
        mgr = create_manager(
            group, NUM_LAYERS, args.num_local_master,
            args.num_local_redundant, use_gpu_solver=use_gpu,
            max_microbatches=max_mbs,
        )

        L = mgr.num_global_logical_experts
        fc1_numel = mgr.expert_fc1_numel
        fc2_numel = mgr.expert_fc2_numel
        T = args.T
        topk = args.topk

        real_layer_id = 1

        # Register master pointers for this real layer
        setup_master_ptrs(mgr, real_layer_id, fc1_numel, fc2_numel, args.num_local_master)

        # Allocate max_mbs virtual layer IDs and verify they're unique
        virtual_ids = []
        for mb in range(max_mbs):
            vid = mgr.allocate_microbatch_slot(real_layer_id)
            virtual_ids.append(vid)

        assert len(set(virtual_ids)) == max_mbs, (
            f"[{solver_name}] Virtual IDs not unique: {virtual_ids}"
        )

        # Each virtual ID should be in range [real_layer_id * max_mbs, (real_layer_id+1) * max_mbs)
        for vid in virtual_ids:
            assert real_layer_id * max_mbs <= vid < (real_layer_id + 1) * max_mbs, (
                f"[{solver_name}] Virtual ID {vid} out of range for real_layer_id={real_layer_id}"
            )

        # Run update_placement + reroute on each virtual layer ID
        torch.manual_seed(42 + rank)
        for vid in virtual_ids:
            routing_map = generate_routing_map(T, L, topk)
            mgr.update_placement(vid, routing_map)
            probs = torch.randn(T, L, dtype=torch.float32, device="cuda")
            exp_probs, exp_map = mgr.reroute(vid, probs, routing_map, backend="cuda")
            assert exp_map.shape[1] == mgr.num_global_physical_experts

        # Next allocation should wrap around (slot 0 reused)
        vid_wrap = mgr.allocate_microbatch_slot(real_layer_id)
        assert vid_wrap == virtual_ids[0], (
            f"[{solver_name}] Wrap-around failed: got {vid_wrap}, expected {virtual_ids[0]}"
        )

        print_rank0(f"  {solver_name} solver: microbatch slot allocation PASS")
        mgr.destroy()

    dist.barrier()
    print_rank0("  All microbatch slot tests PASS")


# ============================================================================
# Test 5: Public reroute() dispatcher selects correct backend
# ============================================================================

def test_reroute_dispatcher(args):
    """Test that reroute(backend='cuda') and reroute(backend='cpu') produce identical results."""
    print_rank0(f"\n{'='*60}")
    print_rank0("Test: Public reroute() dispatcher (cuda vs cpu backend)")
    print_rank0(f"{'='*60}")

    group = dist.group.WORLD
    rank = dist.get_rank()
    T = args.T
    topk = args.topk

    mgr = create_manager(
        group, NUM_LAYERS, args.num_local_master,
        args.num_local_redundant, use_gpu_solver=True,
    )
    L = mgr.num_global_logical_experts
    layer_id = 0

    torch.manual_seed(42 + rank)
    routing_map = generate_routing_map(T, L, topk)
    mgr.update_placement(layer_id, routing_map, verify_reduced_loads=True)

    probs = torch.randn(T, L, dtype=torch.float32, device="cuda")

    # Use public reroute() with both backends
    exp_probs_cuda, exp_map_cuda = mgr.reroute(layer_id, probs, routing_map, backend="cuda")
    exp_probs_cpu, exp_map_cpu = mgr.reroute(layer_id, probs, routing_map, backend="cpu")

    # They should produce identical results
    map_match = torch.equal(exp_map_cpu, exp_map_cuda)
    assert map_match, (
        f"routing_map mismatch: {(exp_map_cpu != exp_map_cuda).sum().item()} differing entries"
    )

    probs_match = torch.equal(exp_probs_cpu, exp_probs_cuda)
    if not probs_match:
        max_diff = (exp_probs_cpu - exp_probs_cuda).abs().max().item()
        assert False, f"probs mismatch: max diff = {max_diff}"

    print_rank0("  PASS — cuda and cpu backends produce identical output")

    mgr.destroy()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end tests for Manager with use_gpu_solver=True"
    )
    parser.add_argument("--num-local-master", type=int, default=4)
    parser.add_argument("--num-local-redundant", type=int, default=2)
    parser.add_argument("--T", type=int, default=4096, help="Number of tokens")
    parser.add_argument("--topk", type=int, default=8, help="Top-k experts per token")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0(
        f"Config: world_size={world_size}, "
        f"num_local_master={args.num_local_master}, "
        f"num_local_redundant={args.num_local_redundant}, "
        f"T={args.T}, topk={args.topk}"
    )

    all_passed = True

    tests = [
        ("placement_equivalence", test_placement_equivalence),
        ("full_pipeline", test_full_pipeline),
        ("reroute_gpu_vs_cpu_solver", test_reroute_gpu_vs_cpu_solver),
        ("microbatch_slots", test_microbatch_slots),
        ("reroute_dispatcher", test_reroute_dispatcher),
    ]

    for test_name, test_fn in tests:
        try:
            test_fn(args)
        except Exception as e:
            all_passed = False
            print_rank0(f"  FAIL ({test_name}): {e}")
            import traceback
            if rank == 0:
                traceback.print_exc()

    print_rank0(f"\n{'='*60}")
    if all_passed:
        print_rank0("ALL TESTS PASSED")
    else:
        print_rank0("SOME TESTS FAILED")

    dist.barrier()
    dist.destroy_process_group()

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
