import argparse
import os
import sys
from typing import Dict, List

import torch
import torch.distributed as dist

import ultra_ep

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import bench, bench_kineto, generate_routing_map_from_distribution

NUM_LAYERS = 48
WEIGHT_ELEMENT_BYTES = 2


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_distribution(distribution: str):
    if distribution not in ("uniform", "skewed", "zipf", "single_hot"):
        raise ValueError(
            f"Unsupported distribution: {distribution}. "
            "Expected one of: uniform, skewed, zipf, single_hot"
        )


def normalize_plan_mode(plan_mode: str) -> str:
    normalized = plan_mode.strip().lower()
    if normalized not in ("direct", "adaptive", "force_relay"):
        raise ValueError(
            "weight_sync_plan_modes entries must be one of: direct, adaptive, force_relay"
        )
    return normalized


def floor_sqrt_int(x: int) -> int:
    root = 0
    while (root + 1) * (root + 1) <= x:
        root += 1
    return root


def choose_relay_count(num_replicas: int, max_relays: int) -> int:
    if num_replicas <= 1:
        return 0
    relay_count = max(1, floor_sqrt_int(num_replicas))
    relay_count = min(relay_count, max_relays)
    return min(relay_count, num_replicas - 1)


def relay_child_count(num_replicas: int, relay_count: int, relay_idx: int) -> int:
    leaf_count = num_replicas - relay_count
    if (
        leaf_count <= 0
        or relay_idx < 0
        or relay_idx >= relay_count
        or relay_idx >= leaf_count
    ):
        return 0
    return (leaf_count - relay_idx + relay_count - 1) // relay_count


def should_use_relay(
    num_replicas: int,
    plan_mode: str,
    relay_min_replicas: int,
    relay_max_relays: int,
    relay_min_fanout_gain: int,
) -> bool:
    if plan_mode == "direct":
        return False

    relay_count = choose_relay_count(num_replicas, relay_max_relays)
    if relay_count <= 0:
        return False

    if plan_mode == "force_relay":
        return True

    if num_replicas < relay_min_replicas:
        return False

    relay_critical_fanout = max(
        relay_count, (num_replicas - relay_count + relay_count - 1) // relay_count
    )
    return (num_replicas - relay_critical_fanout) >= relay_min_fanout_gain


def create_manager(args, plan_mode: str):
    return ultra_ep.Manager(
        group=dist.group.WORLD,
        num_layers=NUM_LAYERS,
        num_local_master_experts=args.num_local_master_experts,
        num_local_redundant_experts=args.num_local_redundant_experts,
        expert_fc1_numel=args.expert_fc1_numel,
        expert_fc2_numel=args.expert_fc2_numel,
        explicitly_destroy=True,
        use_gpu_solver=args.gpu_solver,
        use_quota_eplb_solver=False,
        weight_sync_plan_mode=plan_mode,
        weight_sync_relay_min_replicas=args.weight_sync_relay_min_replicas,
        weight_sync_relay_max_relays=args.weight_sync_relay_max_relays,
        weight_sync_relay_min_fanout_gain=args.weight_sync_relay_min_fanout_gain,
    )


def apply_test_placement(manager, args, layer_id: int, distribution: str):
    num_global_logical_experts = manager.num_global_logical_experts
    assert (
        args.topk <= num_global_logical_experts
    ), f"topk={args.topk} exceeds num_global_logical_experts={num_global_logical_experts}"

    routing_map = generate_routing_map_from_distribution(
        num_tokens=args.num_tokens,
        num_global_logical_experts=num_global_logical_experts,
        topk=args.topk,
        distribution=distribution,
        seed=args.seed + dist.get_rank(),
        num_ranks=dist.get_world_size(),
        num_local_master=args.num_local_master_experts,
        num_nvl_ranks=manager.nvl_domain_size,
        hot_expert_ratio_per_nvl_domain=args.hot_expert_ratio_per_nvl_domain,
        zipf_alpha=args.zipf_alpha,
        single_hot_ratio=args.single_hot_ratio,
    )
    manager.update_placement(layer_id, routing_map, verify_reduced_loads=True)


def build_expected_replica_weights(
    manager,
    args,
    layer_id: int,
    fc1_weights: List[torch.Tensor],
    fc2_weights: List[torch.Tensor],
    replica_weight_buffer_before_sync: torch.Tensor,
) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected_replica_weights = replica_weight_buffer_before_sync.clone()

    local_master_fc1_weights = torch.stack(fc1_weights)
    local_master_fc2_weights = torch.stack(fc2_weights)

    global_master_fc1_weights_list = [
        torch.zeros_like(local_master_fc1_weights) for _ in range(world_size)
    ]
    global_master_fc2_weights_list = [
        torch.zeros_like(local_master_fc2_weights) for _ in range(world_size)
    ]
    dist.all_gather(global_master_fc1_weights_list, local_master_fc1_weights)
    dist.all_gather(global_master_fc2_weights_list, local_master_fc2_weights)

    global_master_fc1_weights = torch.stack(global_master_fc1_weights_list)
    global_master_fc2_weights = torch.stack(global_master_fc2_weights_list)

    for local_replica_idx in range(args.num_local_redundant_experts):
        local_phys_idx = args.num_local_master_experts + local_replica_idx
        global_phys_idx = rank * manager.num_local_physical_experts + local_phys_idx
        logical_idx = manager.physical_to_logical_map[layer_id, global_phys_idx].item()
        if logical_idx < 0:
            continue

        master_global_phys_idx = manager.logical_to_physical_map[
            layer_id, logical_idx, 0
        ].item()
        master_rank = master_global_phys_idx // manager.num_local_physical_experts
        master_local_idx = master_global_phys_idx % manager.num_local_physical_experts

        expected_replica_weights[local_replica_idx, : args.expert_fc1_numel] = (
            global_master_fc1_weights[master_rank, master_local_idx, :]
        )
        expected_replica_weights[local_replica_idx, args.expert_fc1_numel :] = (
            global_master_fc2_weights[master_rank, master_local_idx, :]
        )

    return expected_replica_weights


def analyze_weight_sync_plan(
    manager, args, layer_id: int, plan_mode: str
) -> Dict[str, object]:
    world_size = manager.num_ranks
    weight_bytes_per_expert = manager.expert_total_numel * WEIGHT_ELEMENT_BYTES
    direct_sender_bytes = [0.0 for _ in range(world_size)]
    direct_max_fanout = [0 for _ in range(world_size)]
    plan_stage1_bytes = [0.0 for _ in range(world_size)]
    plan_stage2_bytes = [0.0 for _ in range(world_size)]
    plan_stage1_max_fanout = [0 for _ in range(world_size)]
    plan_stage2_max_fanout = [0 for _ in range(world_size)]
    relay_logical_experts = 0

    for rank in range(world_size):
        for local_master_idx in range(manager.num_local_master_experts):
            global_phys_idx = (
                rank * manager.num_local_physical_experts + local_master_idx
            )
            logical_idx = manager.physical_to_logical_map[
                layer_id, global_phys_idx
            ].item()
            num_replicas = (
                manager.logical_replica_counts[layer_id, logical_idx].item() - 1
            )
            if num_replicas <= 0:
                continue

            direct_sender_bytes[rank] += num_replicas * weight_bytes_per_expert
            direct_max_fanout[rank] = max(direct_max_fanout[rank], num_replicas)

            if should_use_relay(
                num_replicas,
                plan_mode,
                args.weight_sync_relay_min_replicas,
                args.weight_sync_relay_max_relays,
                args.weight_sync_relay_min_fanout_gain,
            ):
                relay_logical_experts += 1
                relay_count = choose_relay_count(
                    num_replicas, args.weight_sync_relay_max_relays
                )
                plan_stage1_bytes[rank] += relay_count * weight_bytes_per_expert
                plan_stage1_max_fanout[rank] = max(
                    plan_stage1_max_fanout[rank], relay_count
                )
                for relay_idx in range(relay_count):
                    relay_global_phys_idx = manager.logical_to_physical_map[
                        layer_id, logical_idx, relay_idx + 1
                    ].item()
                    relay_rank = (
                        relay_global_phys_idx // manager.num_local_physical_experts
                    )
                    child_count = relay_child_count(
                        num_replicas, relay_count, relay_idx
                    )
                    if child_count <= 0:
                        continue
                    plan_stage2_bytes[relay_rank] += (
                        child_count * weight_bytes_per_expert
                    )
                    plan_stage2_max_fanout[relay_rank] = max(
                        plan_stage2_max_fanout[relay_rank], child_count
                    )
            else:
                plan_stage1_bytes[rank] += num_replicas * weight_bytes_per_expert
                plan_stage1_max_fanout[rank] = max(
                    plan_stage1_max_fanout[rank], num_replicas
                )

    flat_critical_bytes = max(direct_sender_bytes) if direct_sender_bytes else 0.0
    relay_stage1_critical_bytes = max(plan_stage1_bytes) if plan_stage1_bytes else 0.0
    relay_stage2_critical_bytes = max(plan_stage2_bytes) if plan_stage2_bytes else 0.0
    relay_serial_critical_bytes = (
        relay_stage1_critical_bytes + relay_stage2_critical_bytes
    )
    relay_overlap_critical_bytes = max(
        relay_stage1_critical_bytes, relay_stage2_critical_bytes
    )
    serial_upper_bound_speedup = (
        flat_critical_bytes / relay_serial_critical_bytes
        if relay_serial_critical_bytes > 0
        else 1.0
    )
    overlap_upper_bound_speedup = (
        flat_critical_bytes / relay_overlap_critical_bytes
        if relay_overlap_critical_bytes > 0
        else 1.0
    )

    return {
        "relay_logical_experts": relay_logical_experts,
        "direct_sender_bytes": direct_sender_bytes,
        "direct_max_fanout": direct_max_fanout,
        "plan_stage1_bytes": plan_stage1_bytes,
        "plan_stage2_bytes": plan_stage2_bytes,
        "plan_total_sender_bytes": [
            stage1 + stage2
            for stage1, stage2 in zip(plan_stage1_bytes, plan_stage2_bytes)
        ],
        "plan_stage1_max_fanout": plan_stage1_max_fanout,
        "plan_stage2_max_fanout": plan_stage2_max_fanout,
        "flat_critical_bytes": flat_critical_bytes,
        "relay_stage1_critical_bytes": relay_stage1_critical_bytes,
        "relay_stage2_critical_bytes": relay_stage2_critical_bytes,
        "relay_serial_critical_bytes": relay_serial_critical_bytes,
        "relay_overlap_critical_bytes": relay_overlap_critical_bytes,
        "serial_upper_bound_speedup": serial_upper_bound_speedup,
        "overlap_upper_bound_speedup": overlap_upper_bound_speedup,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-local-master-experts", type=int, default=4)
    parser.add_argument("--num-local-redundant-experts", type=int, default=2)
    parser.add_argument("--expert-fc1-numel", type=int, default=3072 * 4096)
    parser.add_argument("--expert-fc2-numel", type=int, default=1536 * 4096)
    parser.add_argument("--gpu-solver", action="store_true")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--hot-expert-ratio-per-nvl-domain", type=float, default=0.03)
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--single-hot-ratio", type=float, default=0.8)
    parser.add_argument(
        "--distributions",
        type=str,
        default="uniform,skewed,zipf,single_hot",
    )
    parser.add_argument(
        "--weight-sync-plan-modes",
        type=str,
        default="direct,adaptive,force_relay",
    )
    parser.add_argument("--weight-sync-relay-min-replicas", type=int, default=6)
    parser.add_argument("--weight-sync-relay-max-relays", type=int, default=8)
    parser.add_argument("--weight-sync-relay-min-fanout-gain", type=int, default=2)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--correct-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    distributions = parse_csv_list(args.distributions)
    for distribution in distributions:
        validate_distribution(distribution)
    plan_modes = [
        normalize_plan_mode(mode)
        for mode in parse_csv_list(args.weight_sync_plan_modes)
    ]

    num_nvl_ranks = None
    ranks_to_print = None

    def print_on_leader_ranks(msg: str):
        if rank in ranks_to_print:
            print(msg, flush=True)

    for plan_mode in plan_modes:
        manager = create_manager(args, plan_mode)
        if num_nvl_ranks is None:
            num_nvl_ranks = manager.nvl_domain_size
            ranks_to_print = [
                nvl_domain * num_nvl_ranks
                for nvl_domain in range(world_size // num_nvl_ranks)
            ]
            print_on_leader_ranks(
                f"Running weight_sync test with {world_size} ranks, {num_nvl_ranks} NVL ranks"
            )
            print_on_leader_ranks(
                f"Local experts: {args.num_local_master_experts} master, "
                f"{args.num_local_redundant_experts} redundant"
            )
            print_on_leader_ranks(
                f"Numel: FC1={args.expert_fc1_numel}, FC2={args.expert_fc2_numel}"
            )
            print_on_leader_ranks(
                f"Placement source: Manager.update_placement(use_gpu_solver={args.gpu_solver})"
            )

        print_on_leader_ranks("=" * 80)
        print_on_leader_ranks(f"Weight sync plan mode: {plan_mode}")
        print_on_leader_ranks("=" * 80)

        for distribution in distributions:
            print_on_leader_ranks("-" * 80)
            print_on_leader_ranks(f"Distribution: {distribution}")
            print_on_leader_ranks("-" * 80)

            layer_id = 3
            apply_test_placement(manager, args, layer_id, distribution)

            total_replicas = 0
            max_replicas = 0
            for i in range(args.num_local_master_experts):
                global_phys_idx = rank * manager.num_local_physical_experts + i
                global_log_idx = manager.physical_to_logical_map[
                    layer_id, global_phys_idx
                ].item()
                num_replicas = (
                    manager.logical_replica_counts[layer_id, global_log_idx].item() - 1
                )
                total_replicas += num_replicas
                max_replicas = max(max_replicas, num_replicas)

            total_replicas_tensor = torch.tensor([total_replicas], device="cuda")
            max_replicas_tensor = torch.tensor([max_replicas], device="cuda")
            dist.all_reduce(total_replicas_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_replicas_tensor, op=dist.ReduceOp.MAX)

            plan_metrics = analyze_weight_sync_plan(manager, args, layer_id, plan_mode)
            print_on_leader_ranks(
                f"Total replica tasks across all ranks: {total_replicas_tensor.item()}"
            )
            print_on_leader_ranks(
                f"Max replicas per master: {max_replicas_tensor.item()}"
            )
            print_on_leader_ranks(
                f"Relay-enabled logical experts: {plan_metrics['relay_logical_experts']}"
            )
            print_on_leader_ranks(
                f"Idealized critical sender bytes: flat={plan_metrics['flat_critical_bytes'] / (1024**3):.3f} GB, "
                f"relay-stage1={plan_metrics['relay_stage1_critical_bytes'] / (1024**3):.3f} GB, "
                f"relay-stage2={plan_metrics['relay_stage2_critical_bytes'] / (1024**3):.3f} GB"
            )
            print_on_leader_ranks(
                f"Idealized relay upper bound: serial={plan_metrics['serial_upper_bound_speedup']:.2f}x, "
                f"best-case-overlap={plan_metrics['overlap_upper_bound_speedup']:.2f}x"
            )

            fc1_weights = [
                torch.randn(args.expert_fc1_numel, device="cuda", dtype=torch.bfloat16)
                for _ in range(args.num_local_master_experts)
            ]
            fc2_weights = [
                torch.randn(args.expert_fc2_numel, device="cuda", dtype=torch.bfloat16)
                for _ in range(args.num_local_master_experts)
            ]
            fc1_grads = [
                torch.empty(0, device="cuda", dtype=torch.float32)
                for _ in range(args.num_local_master_experts)
            ]
            fc2_grads = [
                torch.empty(0, device="cuda", dtype=torch.float32)
                for _ in range(args.num_local_master_experts)
            ]
            manager.construct_local_master_ptr_pool(
                layer_id, fc1_weights, fc2_weights, fc1_grads, fc2_grads
            )

            replica_weight_buffer = manager.local_replica_weight_buffer
            replica_weight_buffer.random_(0, 3)
            replica_weight_buffer_before_sync = replica_weight_buffer.clone()
            expected_replica_weights = build_expected_replica_weights(
                manager,
                args,
                layer_id,
                fc1_weights,
                fc2_weights,
                replica_weight_buffer_before_sync,
            )

            dist.barrier()
            manager.weight_sync(layer_id, async_finish=False)
            dist.barrier()

            match = torch.allclose(
                replica_weight_buffer,
                expected_replica_weights,
                atol=args.correct_tolerance,
            )
            assert match, (
                f"Weight sync verification failed on rank {rank} "
                f"(distribution={distribution}, plan_mode={plan_mode})"
            )

            torch.cuda.synchronize()
            dist.barrier()
            print_on_leader_ranks("*** Correctness verification passed! ***")

            def weight_sync_fn():
                manager.weight_sync(layer_id, async_finish=False)

            def pre_fn():
                replica_weight_buffer.random_(0, 3)

            def weight_sync_fn_full():
                replica_weight_buffer.random_(0, 3)
                manager.weight_sync(layer_id, async_finish=False)

            avg_time, min_time, max_time = bench(
                weight_sync_fn,
                num_warmups=args.warmup_iters,
                num_tests=args.bench_iters,
                use_barrier=True,
                pre_fn=pre_fn,
            )
            avg_time_ms = avg_time * 1000
            min_time_ms = min_time * 1000
            max_time_ms = max_time * 1000

            kernel_names = ("weight_sync_kernel",)
            kernel_durations = bench_kineto(
                weight_sync_fn_full,
                kernel_names=kernel_names,
                num_tests=args.bench_iters,
                barrier_comm_profiling=True,
            )

            kernel_dur_second_this_rank = kernel_durations[0]
            kernel_dur_ms_this_rank = kernel_dur_second_this_rank * 1000
            bytes_sent_this_rank = plan_metrics["plan_total_sender_bytes"][rank]
            data_sent_gb_this_rank = bytes_sent_this_rank / (1024**3)
            bandwidth_gbps_this_rank = (
                data_sent_gb_this_rank / kernel_dur_second_this_rank
                if kernel_dur_second_this_rank > 1e-6
                else 0
            )
            print(
                f"[Rank {rank}] kernel duration: {kernel_dur_ms_this_rank:.3f} ms, "
                f"total sent: {data_sent_gb_this_rank:.3f} GB, "
                f"bandwidth: {bandwidth_gbps_this_rank:.2f} GB/s",
                flush=True,
            )
            dist.barrier()

            avg_data_bytes = sum(plan_metrics["plan_total_sender_bytes"]) / world_size
            avg_data_gb = avg_data_bytes / (1024**3)
            avg_bandwidth_gbps = (
                avg_data_gb / (avg_time_ms / 1000.0) if avg_time_ms > 1e-6 else 0.0
            )

            print_on_leader_ranks("Performance metrics:")
            print_on_leader_ranks(
                f"  - E2E Latency: {avg_time_ms:.3f} ms (avg) | "
                f"{min_time_ms:.3f} ms (min) | {max_time_ms:.3f} ms (max)"
            )
            print_on_leader_ranks(f"  - Average Sender Traffic: {avg_data_gb:.3f} GB")
            print_on_leader_ranks(
                f"  - Average Sender Bandwidth: {avg_bandwidth_gbps:.2f} GB/s"
            )
            print_on_leader_ranks(
                f"  - Max Fan-out: flat={max(plan_metrics['direct_max_fanout'])}, "
                f"stage1={max(plan_metrics['plan_stage1_max_fanout'])}, "
                f"stage2={max(plan_metrics['plan_stage2_max_fanout'])}\n"
            )

        manager.destroy()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
