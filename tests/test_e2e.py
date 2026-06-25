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
    nvl_domain_physical_lower_bound,
    print_metric,
    print_section,
    rank_token_count,
)

HYBRIDEP_DISPATCH_KERNEL_NAMES = (
    "scan",
    "permute_preprocessing_kernel",
    "update_expected_value_kernel",
    "device_sync_kernel",
    "dispatch_kernel",
    "permute_kernel",
)
HYBRIDEP_COMBINE_KERNEL_NAMES = (
    "update_expected_value_kernel",
    "device_sync_kernel",
    "combine_kernel",
    "unpermute_kernel",
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


def dtype_for_element_bytes(num_bytes: int) -> torch.dtype:
    mapping = {
        1: torch.uint8,
        2: torch.bfloat16,
        4: torch.float32,
        8: torch.float64,
    }
    if num_bytes not in mapping:
        raise ValueError("Supported test element byte sizes are: 1, 2, 4, 8")
    return mapping[num_bytes]


def randomize_tensor(tensor: torch.Tensor):
    if tensor.numel() == 0:
        return
    tensor.view(torch.uint8).random_(0, 256)


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
        weight_data_dtype=args.weight_data_dtype,
        weight_scale_dtype=args.weight_scale_dtype,
        expert_fc1_weight_scale_numel=args.expert_fc1_weight_scale_numel,
        expert_fc2_weight_scale_numel=args.expert_fc2_weight_scale_numel,
        grad_dtype=args.grad_dtype,
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
    probs = torch.zeros(
        routing_map.shape, dtype=torch.float32, device=routing_map.device
    )
    probs[routing_map] = torch.rand(
        int(routing_map.sum().item()), device=routing_map.device
    )
    return probs


def expected_replica_weights(
    manager, args, layer_id, before_data, before_fc1_scales, before_fc2_scales
):
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
    expected_data = before_data.clone()
    expected_fc1_scales = before_fc1_scales.clone()
    expected_fc2_scales = before_fc2_scales.clone()
    if not bool(valid.any().item()):
        return expected_data, expected_fc1_scales, expected_fc2_scales

    master_phys = manager.logical_to_physical_map[layer_id, logical.clamp_min(0), 0]
    for local_replica_idx in torch.nonzero(valid, as_tuple=False).flatten().tolist():
        gid = int(master_phys[local_replica_idx].item())
        expected_data[local_replica_idx, : args.expert_fc1_numel] = (
            deterministic_tensor(gid, args.expert_fc1_numel, args.weight_data_dtype)
        )
        expected_data[local_replica_idx, args.expert_fc1_numel :] = (
            deterministic_tensor(
                gid + manager.num_global_physical_experts,
                args.expert_fc2_numel,
                args.weight_data_dtype,
            )
        )
        if args.expert_weight_scale_total_numel > 0:
            expected_fc1_scales[local_replica_idx] = deterministic_tensor(
                gid + 2 * manager.num_global_physical_experts,
                args.expert_fc1_weight_scale_numel,
                args.weight_scale_dtype,
            )
            expected_fc2_scales[local_replica_idx] = deterministic_tensor(
                gid + 3 * manager.num_global_physical_experts,
                args.expert_fc2_weight_scale_numel,
                args.weight_scale_dtype,
            )
    return expected_data, expected_fc1_scales, expected_fc2_scales


def expected_master_grads(manager, args, layer_id, fc1_grads, fc2_grads):
    rank = dist.get_rank()
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
    update_avg, _, _ = bench(
        update, args.warmup_iters, args.bench_iters, use_barrier=True
    )
    solve_kernel = bench_kineto(
        update,
        "quota_placement_solve_kernel",
        num_tests=max(3, min(args.bench_iters, 30)),
        barrier_comm_profiling=True,
        suppress_kineto_output=True,
    )

    expanded_probs, expanded_routing = manager.reroute(layer_id, probs, routing_map)
    assert bool((expanded_routing.sum(dim=1) == args.topk).all().item())
    rank_load_before = (
        routing_map.sum(dim=0, dtype=torch.int32)
        .view(dist.get_world_size(), args.num_local_master)
        .sum(dim=1)
    )
    rank_load_after = (
        expanded_routing.sum(dim=0, dtype=torch.int32)
        .view(dist.get_world_size(), manager.num_local_physical_experts)
        .sum(dim=1)
    )
    dist.all_reduce(rank_load_before)
    dist.all_reduce(rank_load_after)
    physical_lower_bound = nvl_domain_physical_lower_bound(
        rank_load_before, manager.nvl_domain_size
    )

    def reroute():
        manager.reroute(layer_id, probs, routing_map)

    reroute_avg, _, _ = bench(
        reroute, args.warmup_iters, args.bench_iters, use_barrier=True
    )
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
        f"{max_mean(rank_load_after).item():.3f} "
        f"(lower bound {physical_lower_bound.item():.3f})",
        print_fn=print_rank0,
    )
    return expanded_probs, expanded_routing


def run_weight_sync(manager, args, layer_id, plan_mode: str):
    manager.set_weight_sync_plan_mode(plan_mode)
    fc1_weights = [
        deterministic_tensor(
            dist.get_rank() * manager.num_local_physical_experts + local_idx,
            args.expert_fc1_numel,
            args.weight_data_dtype,
        )
        for local_idx in range(args.num_local_master)
    ]
    fc2_weights = [
        deterministic_tensor(
            dist.get_rank() * manager.num_local_physical_experts
            + local_idx
            + manager.num_global_physical_experts,
            args.expert_fc2_numel,
            args.weight_data_dtype,
        )
        for local_idx in range(args.num_local_master)
    ]
    fc1_scales = None
    fc2_scales = None
    if args.expert_weight_scale_total_numel > 0:
        fc1_scales = [
            deterministic_tensor(
                dist.get_rank() * manager.num_local_physical_experts
                + local_idx
                + 2 * manager.num_global_physical_experts,
                args.expert_fc1_weight_scale_numel,
                args.weight_scale_dtype,
            )
            for local_idx in range(args.num_local_master)
        ]
        fc2_scales = [
            deterministic_tensor(
                dist.get_rank() * manager.num_local_physical_experts
                + local_idx
                + 3 * manager.num_global_physical_experts,
                args.expert_fc2_weight_scale_numel,
                args.weight_scale_dtype,
            )
            for local_idx in range(args.num_local_master)
        ]
    dummy_grads = [
        torch.empty(0, device="cuda", dtype=args.grad_dtype)
        for _ in range(args.num_local_master)
    ]
    manager.construct_local_master_ptr_pool(
        layer_id,
        fc1_weights,
        fc2_weights,
        dummy_grads,
        dummy_grads,
        fc1_weight_scales=fc1_scales,
        fc2_weight_scales=fc2_scales,
    )
    replica = manager.local_replica_weight_buffer
    assert bitwise_equal(
        manager.local_replica_fc1_weight_buffer, replica[:, : args.expert_fc1_numel]
    )
    assert bitwise_equal(
        manager.local_replica_fc2_weight_buffer, replica[:, args.expert_fc1_numel :]
    )
    replica_fc1_scales = manager.local_replica_fc1_weight_scale_buffer
    replica_fc2_scales = manager.local_replica_fc2_weight_scale_buffer
    randomize_tensor(replica)
    randomize_tensor(replica_fc1_scales)
    randomize_tensor(replica_fc2_scales)
    expected, expected_fc1_scales, expected_fc2_scales = expected_replica_weights(
        manager,
        args,
        layer_id,
        replica.clone(),
        replica_fc1_scales.clone(),
        replica_fc2_scales.clone(),
    )
    dist.barrier()
    manager.weight_sync(layer_id, async_finish=False)
    torch.cuda.synchronize()
    dist.barrier()
    data_ok = bitwise_equal(replica, expected)
    scales_ok = bitwise_equal(
        replica_fc1_scales, expected_fc1_scales
    ) and bitwise_equal(replica_fc2_scales, expected_fc2_scales)
    assert (
        data_ok and scales_ok
    ), f"weight_sync bitwise mismatch on rank {dist.get_rank()}"

    def sync_once():
        manager.weight_sync(layer_id, async_finish=False)

    def randomize_replica():
        randomize_tensor(replica)
        randomize_tensor(replica_fc1_scales)
        randomize_tensor(replica_fc2_scales)

    avg, _, _ = bench(
        sync_once,
        args.warmup_iters,
        args.bench_iters,
        use_barrier=True,
        pre_fn=randomize_replica,
    )
    kernel_parts = bench_kineto(
        sync_once,
        ("weight_sync_kernel", "weight_sync_thread_copy_kernel"),
        num_tests=max(3, min(args.bench_iters, 30)),
        barrier_comm_profiling=True,
        suppress_kineto_output=True,
    )
    kernel = sum(kernel_parts)
    detail = f"{args.weight_data_bytes}B data bitwise PASS"
    if args.expert_weight_scale_total_numel > 0:
        detail += f", {args.weight_scale_bytes}B scales bitwise PASS"
    print_metric(
        f"weight_sync/{plan_mode}",
        avg * 1000,
        kernel * 1000,
        detail,
        print_fn=print_rank0,
    )


def run_grad_reduce(manager, args, layer_id, deterministic: bool):
    manager.set_grad_reduce_deterministic(deterministic)
    fc1_weights = [
        torch.empty(0, device="cuda", dtype=args.weight_data_dtype)
        for _ in range(args.num_local_master)
    ]
    fc2_weights = [
        torch.empty(0, device="cuda", dtype=args.weight_data_dtype)
        for _ in range(args.num_local_master)
    ]
    fc1_scales = fc2_scales = None
    if args.expert_weight_scale_total_numel > 0:
        fc1_scales = [
            torch.empty(0, device="cuda", dtype=args.weight_scale_dtype)
            for _ in range(args.num_local_master)
        ]
        fc2_scales = [
            torch.empty(0, device="cuda", dtype=args.weight_scale_dtype)
            for _ in range(args.num_local_master)
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
    manager.construct_local_master_ptr_pool(
        layer_id,
        fc1_weights,
        fc2_weights,
        fc1_grads,
        fc2_grads,
        fc1_weight_scales=fc1_scales,
        fc2_weight_scales=fc2_scales,
    )
    replica = manager.local_replica_grad_buffer
    assert bitwise_equal(
        manager.local_replica_fc1_grad_buffer, replica[:, : args.expert_fc1_numel]
    )
    assert bitwise_equal(
        manager.local_replica_fc2_grad_buffer, replica[:, args.expert_fc1_numel :]
    )
    replica_ref = torch.empty_like(replica)
    replica_base = (
        dist.get_rank() * manager.num_local_physical_experts + args.num_local_master
    )
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
        manager, args, layer_id, fc1_grads, fc2_grads
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
    valid_local_replicas = (
        manager.physical_to_logical_map[layer_id, local_replica_phys] >= 0
    )
    zero_correct = True
    if bool(valid_local_replicas.any().item()):
        zero_correct = bool((replica[valid_local_replicas] == 0).all().item())
    assert zero_correct, f"grad_reduce replica zero mismatch on rank {dist.get_rank()}"

    correct = True
    for idx in range(args.num_local_master):
        if deterministic:
            correct = correct and bitwise_equal(fc1_grads[idx], expected_fc1[idx])
            correct = correct and bitwise_equal(fc2_grads[idx], expected_fc2[idx])
        else:
            correct = correct and torch.allclose(
                fc1_grads[idx], expected_fc1[idx], atol=1e-5, rtol=1e-5
            )
            correct = correct and torch.allclose(
                fc2_grads[idx], expected_fc2[idx], atol=1e-5, rtol=1e-5
            )
    check_name = "bitwise" if deterministic else "allclose"
    assert correct, f"grad_reduce {check_name} mismatch on rank {dist.get_rank()}"

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
    kernel_name = (
        "grad_reduce_deterministic_kernel" if deterministic else "grad_reduce_kernel"
    )
    kernel = bench_kineto(
        reduce_once,
        kernel_name,
        num_tests=max(3, min(args.bench_iters, 30)),
        barrier_comm_profiling=True,
        suppress_kineto_output=True,
    )
    mode_name = "deterministic" if deterministic else "atomic"
    print_metric(
        f"grad_reduce/{mode_name}",
        avg * 1000,
        kernel * 1000,
        f"{check_name} PASS ({manager.grad_reduce_num_sms} SMs)",
        print_fn=print_rank0,
    )


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
        padded = torch.zeros(
            max_tokens, routing.size(1), dtype=torch.bool, device="cuda"
        )
        padded[: routing.size(0)] = routing
        padded[routing.size(0) :, rank * num_local_physical] = True
        routing = padded
    probs = routing.float()
    hidden = torch.randn(
        max_tokens, args.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
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
    dispatched, dispatched_probs, _, tokens_per_expert, handle = (
        buffer.dispatch_with_permute(
            hidden=hidden,
            routing_map=routing,
            probs=probs,
            scaling_factor=None,
            pad_multiple=args.pad_multiple,
        )
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

    dispatch_avg, _, _ = bench(
        dispatch_once, args.warmup_iters, args.bench_iters, use_barrier=True
    )
    combine_avg, _, _ = bench(
        combine_once, args.warmup_iters, args.bench_iters, use_barrier=True
    )
    dispatch_kernel_parts = bench_kineto(
        dispatch_once,
        HYBRIDEP_DISPATCH_KERNEL_NAMES,
        num_tests=max(3, min(args.bench_iters, 30)),
        barrier_comm_profiling=True,
        suppress_kineto_output=True,
    )
    combine_kernel_parts = bench_kineto(
        combine_once,
        HYBRIDEP_COMBINE_KERNEL_NAMES,
        num_tests=max(3, min(args.bench_iters, 30)),
        barrier_comm_profiling=True,
        suppress_kineto_output=True,
    )
    dispatch_kernel = sum(dispatch_kernel_parts)
    combine_kernel = sum(combine_kernel_parts)
    print_metric(
        "HybridEP dispatch",
        dispatch_avg * 1000,
        dispatch_kernel * 1000,
        print_fn=print_rank0,
    )
    print_metric(
        "HybridEP combine",
        combine_avg * 1000,
        combine_kernel * 1000,
        print_fn=print_rank0,
    )


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

    _, expanded_routing = run_update_and_reroute(
        manager, args, layer_id, routing_map, probs
    )
    for plan_mode in args.weight_sync_plan_modes:
        run_weight_sync(manager, args, layer_id, plan_mode)
    for deterministic in (False, True):
        run_grad_reduce(manager, args, layer_id, deterministic)
    if args.include_token_a2a:
        run_hybridep_a2a(args, expanded_routing)


def main():
    parser = argparse.ArgumentParser(description="UltraEP distributed e2e test")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--num-redundant-experts-per-rank", type=int, default=2)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tokens-per-rank", type=int, default=8192)
    parser.add_argument(
        "--variable-input-tokens", action="store_true", dest="variable_input_tokens"
    )
    parser.add_argument(
        "--imbalance-ratios",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0],
        help="Space-separated target rank-level max/mean ratios (must be >= 1).",
    )
    parser.add_argument("--expert-fc1-numel", type=int, default=3072 * 4096)
    parser.add_argument("--expert-fc2-numel", type=int, default=1536 * 4096)
    parser.add_argument(
        "--weight-data-bytes",
        type=int,
        default=2,
        help="Expert weight data element bytes used by the byte-copy test.",
    )
    parser.add_argument(
        "--weight-scale-bytes",
        type=int,
        default=4,
        help="Expert weight scale element bytes used when scale numel is non-zero.",
    )
    parser.add_argument("--expert-fc1-weight-scale-numel", type=int, default=0)
    parser.add_argument("--expert-fc2-weight-scale-numel", type=int, default=0)
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
    if args.expert_fc1_weight_scale_numel < 0 or args.expert_fc2_weight_scale_numel < 0:
        raise ValueError("expert weight scale numel must be non-negative")
    args.weight_data_dtype = dtype_for_element_bytes(args.weight_data_bytes)
    args.weight_scale_dtype = dtype_for_element_bytes(args.weight_scale_bytes)
    args.grad_dtype = torch.float32
    args.expert_weight_scale_total_numel = (
        args.expert_fc1_weight_scale_numel + args.expert_fc2_weight_scale_numel
    )

    group = setup_dist()
    manager = create_manager(args, group)
    print_section(
        "UltraEP E2E Test | "
        f"world={dist.get_world_size()} experts={args.num_experts} "
        f"local_master/rank={args.num_local_master} "
        f"redundant/rank={args.num_redundant_experts_per_rank} topk={args.topk} "
        f"tokens/rank<= {args.tokens_per_rank} "
        f"weight_data_bytes={args.weight_data_bytes} "
        f"scales={args.expert_weight_scale_total_numel}x{args.weight_scale_bytes}B",
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
