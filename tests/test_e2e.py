import argparse
import os
import sys

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
if "" in sys.path and os.path.abspath(os.getcwd()) == REPO_ROOT:
    sys.path.remove("")
import ultra_ep

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    bench,
    bench_kineto,
    bitwise_equal,
    expert_load_imbalance_summary,
    format_load_imbalance,
    generate_routing_map_zipf,
    max_mean,
    print_metric,
    print_section,
    rank_token_count,
)


def print_rank0(msg: str):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def replica_summary(manager, layer_id: int) -> str:
    replicas = manager.logical_replica_counts[layer_id].float() - 1.0
    return (
        f"replicas max/mean/min = {replicas.max().item():.0f}/"
        f"{replicas.mean().item():.2f}/{replicas.min().item():.0f}"
    )


def deterministic_tensor(global_id: int | torch.Tensor, numel: int, dtype: torch.dtype):
    idx = torch.arange(numel, device="cuda", dtype=torch.int64)
    if isinstance(global_id, torch.Tensor):
        gid = global_id.to(device="cuda", dtype=torch.int64)
    else:
        gid = torch.tensor(global_id, device="cuda", dtype=torch.int64)
    values = ((idx * 131 + gid * 17) % 2048).to(torch.float32) / 128.0
    return values.to(dtype)


def setup_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return dist.group.WORLD


def create_manager(args, group):
    world_size = dist.get_world_size(group)
    if args.num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by distributed world size")
    args.num_local_master = args.num_experts // world_size
    return ultra_ep.Manager(
        group=group,
        num_layers=1,
        num_local_master_experts=args.num_local_master,
        num_local_redundant_experts=args.num_redundant_experts_per_rank,
        expert_fc1_numel=args.expert_fc1_numel,
        expert_fc2_numel=args.expert_fc2_numel,
        explicitly_destroy=True,
    )


def make_case_routing(args, actual_tokens: int):
    return generate_routing_map_zipf(
        actual_tokens,
        args.num_experts,
        dist.get_world_size(),
        args.num_local_master,
        args.topk,
        args.imbalance_ratio,
        args.seed,
        rank=dist.get_rank(),
    )


def make_probs(routing_map: torch.Tensor):
    probs = torch.zeros(routing_map.shape, dtype=torch.float32, device=routing_map.device)
    probs[routing_map] = torch.rand(int(routing_map.sum().item()), device=routing_map.device)
    return probs


def expected_replica_weights(manager, args, layer_id, fc1_weights, fc2_weights, before):
    rank = dist.get_rank()
    num_local_physical = manager.num_local_physical_experts
    placement_device = manager.physical_to_logical_map.device
    local_replica_phys = (
        rank * num_local_physical
        + args.num_local_master
        + torch.arange(args.num_redundant_experts_per_rank, device=placement_device)
    )
    logical = manager.physical_to_logical_map[layer_id, local_replica_phys]
    valid = logical >= 0
    expected = before.clone()
    if not bool(valid.any().item()):
        return expected

    master_phys = manager.logical_to_physical_map[layer_id, logical.clamp_min(0), 0]
    for local_replica_idx in torch.nonzero(valid, as_tuple=False).flatten().tolist():
        gid = int(master_phys[local_replica_idx].item())
        expected[local_replica_idx, : args.expert_fc1_numel] = deterministic_tensor(
            gid, args.expert_fc1_numel, torch.bfloat16
        )
        expected[local_replica_idx, args.expert_fc1_numel :] = deterministic_tensor(
            gid + manager.num_global_physical_experts,
            args.expert_fc2_numel,
            torch.bfloat16,
        )
    return expected


def expected_master_grads(manager, args, layer_id, fc1_grads, fc2_grads, replica_before):
    rank = dist.get_rank()
    num_local_physical = manager.num_local_physical_experts
    expected_fc1 = [g.clone() for g in fc1_grads]
    expected_fc2 = [g.clone() for g in fc2_grads]
    for local_idx in range(args.num_local_master):
        logical_id = rank * args.num_local_master + local_idx
        row = manager.logical_to_physical_map[layer_id, logical_id]
        count = int(manager.logical_replica_counts[layer_id, logical_id].item())
        for replica_slot in range(1, count):
            phys = int(row[replica_slot].item())
            expected_fc1[local_idx] = expected_fc1[local_idx] + deterministic_tensor(
                phys, args.expert_fc1_numel, torch.float32
            )
            expected_fc2[local_idx] = expected_fc2[local_idx] + deterministic_tensor(
                phys + manager.num_global_physical_experts,
                args.expert_fc2_numel,
                torch.float32,
            )
    return expected_fc1, expected_fc2


def run_update_and_reroute(manager, args, layer_id, routing_map, probs):
    def update():
        manager.update_placement(layer_id, routing_map, verify_reduced_loads=True)

    update()
    update_avg, _, _ = bench(update, args.warmup_iters, args.bench_iters, use_barrier=True)
    solve_kernel = bench_kineto(
        update,
        "quota_placement_solve_kernel",
        num_tests=max(3, min(args.bench_iters, 30)),
        barrier_comm_profiling=True,
        suppress_kineto_output=True,
    )

    expanded_probs, expanded_routing = manager.reroute(layer_id, probs, routing_map)
    assert bool((expanded_routing.sum(dim=1) == args.topk).all().item())
    rank_load_before = routing_map.sum(dim=0, dtype=torch.int32).view(
        dist.get_world_size(), args.num_local_master
    ).sum(dim=1)
    rank_load_after = expanded_routing.sum(dim=0, dtype=torch.int32).view(
        dist.get_world_size(), manager.num_local_physical_experts
    ).sum(dim=1)
    dist.all_reduce(rank_load_before)
    dist.all_reduce(rank_load_after)

    def reroute():
        manager.reroute(layer_id, probs, routing_map)

    reroute_avg, _, _ = bench(reroute, args.warmup_iters, args.bench_iters, use_barrier=True)
    reroute_kernel_parts = bench_kineto(
        reroute,
        ("reroute_forward_count_kernel", "dense_quota_reroute_scatter_kernel"),
        num_tests=max(3, min(args.bench_iters, 30)),
        barrier_comm_profiling=True,
        suppress_kineto_output=True,
    )
    reroute_kernel = sum(reroute_kernel_parts)
    print_metric(
        "update_placement",
        update_avg * 1000,
        solve_kernel * 1000,
        replica_summary(manager, layer_id),
        print_fn=print_rank0,
    )
    print_metric(
        "reroute",
        reroute_avg * 1000,
        reroute_kernel * 1000,
        f"rank max/mean {max_mean(rank_load_before).item():.3f} -> "
        f"{max_mean(rank_load_after).item():.3f}",
        print_fn=print_rank0,
    )
    return expanded_probs, expanded_routing


def run_weight_sync(manager, args, layer_id, plan_mode: str):
    manager.set_weight_sync_plan_mode(plan_mode)
    fc1_weights = [
        deterministic_tensor(
            dist.get_rank() * manager.num_local_physical_experts + local_idx,
            args.expert_fc1_numel,
            torch.bfloat16,
        )
        for local_idx in range(args.num_local_master)
    ]
    fc2_weights = [
        deterministic_tensor(
            dist.get_rank() * manager.num_local_physical_experts
            + local_idx
            + manager.num_global_physical_experts,
            args.expert_fc2_numel,
            torch.bfloat16,
        )
        for local_idx in range(args.num_local_master)
    ]
    dummy_grads = [
        torch.empty(0, device="cuda", dtype=torch.float32) for _ in range(args.num_local_master)
    ]
    manager.construct_local_master_ptr_pool(layer_id, fc1_weights, fc2_weights, dummy_grads, dummy_grads)
    replica = manager.local_replica_weight_buffer
    replica.random_(0, 3)
    expected = expected_replica_weights(manager, args, layer_id, fc1_weights, fc2_weights, replica.clone())
    dist.barrier()
    manager.weight_sync(layer_id, async_finish=False)
    torch.cuda.synchronize()
    dist.barrier()
    assert bitwise_equal(replica, expected), f"weight_sync bitwise mismatch on rank {dist.get_rank()}"

    def sync_once():
        manager.weight_sync(layer_id, async_finish=False)

    def randomize_replica():
        replica.random_(0, 3)

    avg, _, _ = bench(
        sync_once,
        args.warmup_iters,
        args.bench_iters,
        use_barrier=True,
        pre_fn=randomize_replica,
    )
    kernel = bench_kineto(
        sync_once,
        "weight_sync_kernel",
        num_tests=max(3, min(args.bench_iters, 30)),
        barrier_comm_profiling=True,
        suppress_kineto_output=True,
    )
    print_metric(f"weight_sync/{plan_mode}", avg * 1000, kernel * 1000, "bitwise PASS", print_fn=print_rank0)


def run_grad_reduce(manager, args, layer_id):
    fc1_weights = [
        torch.empty(0, device="cuda", dtype=torch.bfloat16) for _ in range(args.num_local_master)
    ]
    fc2_weights = [
        torch.empty(0, device="cuda", dtype=torch.bfloat16) for _ in range(args.num_local_master)
    ]
    fc1_grads = [
        deterministic_tensor(
            dist.get_rank() * manager.num_local_physical_experts + local_idx,
            args.expert_fc1_numel,
            torch.float32,
        )
        for local_idx in range(args.num_local_master)
    ]
    fc2_grads = [
        deterministic_tensor(
            dist.get_rank() * manager.num_local_physical_experts
            + local_idx
            + manager.num_global_physical_experts,
            args.expert_fc2_numel,
            torch.float32,
        )
        for local_idx in range(args.num_local_master)
    ]
    manager.construct_local_master_ptr_pool(layer_id, fc1_weights, fc2_weights, fc1_grads, fc2_grads)
    replica = manager.local_replica_grad_buffer
    replica_ref = torch.empty_like(replica)
    replica_base = dist.get_rank() * manager.num_local_physical_experts + args.num_local_master
    for local_replica_idx in range(args.num_redundant_experts_per_rank):
        phys = replica_base + local_replica_idx
        replica_ref[local_replica_idx, : args.expert_fc1_numel] = deterministic_tensor(
            phys, args.expert_fc1_numel, torch.float32
        )
        replica_ref[local_replica_idx, args.expert_fc1_numel :] = deterministic_tensor(
            phys + manager.num_global_physical_experts,
            args.expert_fc2_numel,
            torch.float32,
        )
    fc1_base = [g.clone() for g in fc1_grads]
    fc2_base = [g.clone() for g in fc2_grads]
    replica.copy_(replica_ref)
    expected_fc1, expected_fc2 = expected_master_grads(
        manager, args, layer_id, fc1_grads, fc2_grads, replica_ref
    )
    dist.barrier()
    manager.grad_reduce(layer_id, async_finish=False)
    torch.cuda.synchronize()
    dist.barrier()
    placement_device = manager.physical_to_logical_map.device
    local_replica_phys = (
        dist.get_rank() * manager.num_local_physical_experts
        + args.num_local_master
        + torch.arange(args.num_redundant_experts_per_rank, device=placement_device)
    )
    valid_local_replicas = manager.physical_to_logical_map[layer_id, local_replica_phys] >= 0
    correct = True
    if bool(valid_local_replicas.any().item()):
        correct = bool((replica[valid_local_replicas] == 0).all().item())
    for idx in range(args.num_local_master):
        correct = correct and torch.allclose(fc1_grads[idx], expected_fc1[idx], atol=1e-5, rtol=1e-5)
        correct = correct and torch.allclose(fc2_grads[idx], expected_fc2[idx], atol=1e-5, rtol=1e-5)

    def reset_grad_state():
        for idx in range(args.num_local_master):
            fc1_grads[idx].copy_(fc1_base[idx])
            fc2_grads[idx].copy_(fc2_base[idx])
        replica.copy_(replica_ref)

    def reduce_once():
        manager.grad_reduce(layer_id, async_finish=False)

    avg, _, _ = bench(
        reduce_once,
        args.warmup_iters,
        args.bench_iters,
        use_barrier=True,
        pre_fn=reset_grad_state,
    )
    kernel = bench_kineto(
        reduce_once,
        "grad_reduce_kernel",
        num_tests=max(3, min(args.bench_iters, 30)),
        barrier_comm_profiling=True,
        suppress_kineto_output=True,
    )
    status = "PASS" if correct else "MISMATCH"
    print_metric("grad_reduce", avg * 1000, kernel * 1000, f"correctness {status}", print_fn=print_rank0)


def run_hybridep_a2a(args, expanded_routing):
    try:
        import deep_ep
    except ImportError:
        print_rank0("HybridEP is not importable; skip token dispatch/combine")
        return

    max_tokens = args.tokens_per_rank
    rank = dist.get_rank()
    num_local_physical = args.num_local_master + args.num_redundant_experts_per_rank
    routing = expanded_routing
    if routing.size(0) < max_tokens:
        padded = torch.zeros(max_tokens, routing.size(1), dtype=torch.bool, device="cuda")
        padded[: routing.size(0)] = routing
        padded[routing.size(0) :, rank * num_local_physical] = True
        routing = padded
    probs = routing.float()
    hidden = torch.randn(max_tokens, args.hidden_size, dtype=torch.bfloat16, device="cuda")
    if expanded_routing.size(0) < max_tokens:
        hidden[expanded_routing.size(0) :] = 0

    buffer = deep_ep.HybridEPBuffer(
        group=dist.group.WORLD,
        hidden_dim=args.hidden_size,
        max_num_of_tokens_per_rank=max_tokens,
        num_local_experts=num_local_physical,
        use_fp8=False,
        num_sms_dispatch_api=args.hybridep_num_sms,
        num_sms_combine_api=args.hybridep_num_sms,
    )
    dispatched, dispatched_probs, _, tokens_per_expert, handle = buffer.dispatch_with_permute(
        hidden=hidden, routing_map=routing, probs=probs, scaling_factor=None, pad_multiple=args.pad_multiple
    )
    num_permuted = int(tokens_per_expert.sum().item())
    combined, _ = buffer.combine_with_unpermute(
        hidden=dispatched.to(torch.bfloat16),
        probs=dispatched_probs,
        handle=handle,
        pad_multiple=args.pad_multiple,
    )
    assert combined.shape == hidden.shape

    def dispatch_once():
        return buffer.dispatch_with_permute(
            hidden=hidden,
            routing_map=routing,
            probs=probs,
            scaling_factor=None,
            pad_multiple=args.pad_multiple,
            num_permuted_tokens=num_permuted,
        )

    def combine_once():
        buffer.combine_with_unpermute(
            hidden=dispatched.to(torch.bfloat16),
            probs=dispatched_probs,
            handle=handle,
            pad_multiple=args.pad_multiple,
        )

    dispatch_avg, _, _ = bench(dispatch_once, args.warmup_iters, args.bench_iters, use_barrier=True)
    combine_avg, _, _ = bench(combine_once, args.warmup_iters, args.bench_iters, use_barrier=True)
    dispatch_kernel, combine_kernel = bench_kineto(
        lambda: (dispatch_once(), combine_once()),
        ("dispatch_kernel", "combine_kernel"),
        num_tests=max(3, min(args.bench_iters, 30)),
        barrier_comm_profiling=True,
        suppress_kineto_output=True,
    )
    print_metric("HybridEP dispatch", dispatch_avg * 1000, dispatch_kernel * 1000, print_fn=print_rank0)
    print_metric("HybridEP combine", combine_avg * 1000, combine_kernel * 1000, print_fn=print_rank0)


def run_case(manager, args, ratio: float):
    args.imbalance_ratio = ratio
    layer_id = 0
    actual_tokens = rank_token_count(
        dist.get_rank(), args.tokens_per_rank, args.variable_input_tokens, args.seed
    )
    routing_map = make_case_routing(args, actual_tokens)
    probs = make_probs(routing_map)
    token_count = torch.tensor([actual_tokens], device="cuda", dtype=torch.int64)
    token_min = token_count.clone()
    token_max = token_count.clone()
    token_sum = token_count.clone()
    dist.all_reduce(token_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(token_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(token_sum, op=dist.ReduceOp.SUM)

    expert_loads = routing_map.sum(dim=0, dtype=torch.int32)
    dist.all_reduce(expert_loads)
    load_summary = expert_load_imbalance_summary(
        expert_loads, dist.get_world_size(), args.num_local_master
    )
    print_section(
        f"Imbalance Ratio {ratio:g} | tokens/rank min/mean/max = "
        f"{int(token_min.item())}/{token_sum.item() / dist.get_world_size():.1f}/"
        f"{int(token_max.item())}\n"
        f"{format_load_imbalance(load_summary)}",
        print_fn=print_rank0,
    )

    _, expanded_routing = run_update_and_reroute(manager, args, layer_id, routing_map, probs)
    for plan_mode in args.weight_sync_plan_modes:
        run_weight_sync(manager, args, layer_id, plan_mode)
    run_grad_reduce(manager, args, layer_id)
    if args.include_token_a2a:
        run_hybridep_a2a(args, expanded_routing)


def main():
    parser = argparse.ArgumentParser(description="UltraEP distributed e2e test")
    parser.add_argument("--num-experts", type=int, default=128)
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
    parser.add_argument("--expert-fc1-numel", type=int, default=3072 * 4096)
    parser.add_argument("--expert-fc2-numel", type=int, default=1536 * 4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--include-token-a2a", action="store_true")
    parser.add_argument(
        "--weight-sync-plan-modes",
        nargs="+",
        default=["direct", "adaptive_relay"],
        choices=["direct", "adaptive_relay", "force_relay"],
        help="Space-separated weight sync plan modes.",
    )
    parser.add_argument("--hybridep-num-sms", type=int, default=24)
    parser.add_argument("--pad-multiple", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()
    if any(ratio < 1.0 for ratio in args.imbalance_ratios):
        raise ValueError("--imbalance-ratios entries must be >= 1")

    group = setup_dist()
    manager = create_manager(args, group)
    print_section(
        "UltraEP E2E Test | "
        f"world={dist.get_world_size()} experts={args.num_experts} "
        f"local_master/rank={args.num_local_master} "
        f"redundant/rank={args.num_redundant_experts_per_rank} topk={args.topk} "
        f"tokens/rank<= {args.tokens_per_rank}",
        print_fn=print_rank0,
    )

    try:
        for ratio in args.imbalance_ratios:
            run_case(manager, args, ratio)
    finally:
        manager.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
