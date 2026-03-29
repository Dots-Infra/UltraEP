import torch
import torch.distributed as dist
import os, sys
import argparse
import ultra_ep
from ultra_ep.util import setup_placement_random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import bench, bench_kineto, generate_routing_map_from_distribution

NUM_LAYERS = 48


def apply_test_placement(manager, args, layer_id: int, placement_mode: str):
    if args.gpu_solver:
        num_global_logical_experts = manager.num_global_logical_experts
        assert (
            args.topk <= num_global_logical_experts
        ), f"topk={args.topk} exceeds num_global_logical_experts={num_global_logical_experts}"
        routing_map = generate_routing_map_from_distribution(
            num_tokens=args.num_tokens,
            num_global_logical_experts=num_global_logical_experts,
            topk=args.topk,
            distribution=placement_mode,
            seed=args.seed + dist.get_rank(),
            num_ranks=dist.get_world_size(),
            num_local_master=args.num_local_master_experts,
            num_nvl_ranks=manager.nvl_domain_size,
            hot_expert_ratio_per_nvl_domain=args.hot_expert_ratio_per_nvl_domain,
        )
        manager.update_placement(layer_id, routing_map, verify_reduced_loads=True)
        return

    world_size = dist.get_world_size()
    setup_placement_random(
        world_size,
        args.num_local_master_experts,
        args.num_local_redundant_experts,
        manager.physical_to_logical_map,
        manager.logical_to_physical_map,
        manager.logical_replica_counts,
        replica_distribution=placement_mode,
        num_nvl_ranks=manager.nvl_domain_size,
        hot_expert_ratio_per_nvl_domain=0.03,
        seed=args.seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-local-master-experts", type=int, default=4)
    parser.add_argument("--num-local-redundant-experts", type=int, default=2)
    parser.add_argument("--expert-fc1-numel", type=int, default=3072 * 4096)
    parser.add_argument("--expert-fc2-numel", type=int, default=1536 * 4096)
    parser.add_argument("--gpu-solver", action="store_true")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--hot-expert-ratio-per-nvl-domain", type=float, default=0.03)
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

    manager = ultra_ep.Manager(
        group=dist.group.WORLD,
        num_layers=NUM_LAYERS,
        num_local_master_experts=args.num_local_master_experts,
        num_local_redundant_experts=args.num_local_redundant_experts,
        expert_fc1_numel=args.expert_fc1_numel,
        expert_fc2_numel=args.expert_fc2_numel,
        explicitly_destroy=True,
        use_gpu_solver=args.gpu_solver,
        use_quota_eplb_solver=False,
    )

    num_nvl_ranks = manager.nvl_domain_size
    ranks_to_print = [
        nvl_domain * num_nvl_ranks for nvl_domain in range(world_size // num_nvl_ranks)
    ]

    def print_on_leader_ranks(msg: str):
        if rank in ranks_to_print:
            print(msg, flush=True)

    print_on_leader_ranks(
        f"Running test with {world_size} ranks, {num_nvl_ranks} NVL ranks"
    )
    print_on_leader_ranks(
        f"Local experts: {args.num_local_master_experts} master, {args.num_local_redundant_experts} redundant"
    )
    print_on_leader_ranks(
        f"Numel: FC1={args.expert_fc1_numel}, FC2={args.expert_fc2_numel}"
    )
    print_on_leader_ranks(
        f"Placement source: {'Manager.update_placement(use_gpu_solver=True)' if args.gpu_solver else 'setup_placement_random()'}"
    )

    for mode in ["low_sm", "high_sm"]:
        for placement_mode in ["uniform", "skewed"]:
            print_on_leader_ranks(f"=" * 80)
            print_on_leader_ranks(
                f"Test expert placement with {mode} mode, {placement_mode} distribution"
            )
            print_on_leader_ranks(f"=" * 80)

            # Prepare data
            layer_id = 3
            apply_test_placement(manager, args, layer_id, placement_mode)
            expert_total_numel = args.expert_fc1_numel + args.expert_fc2_numel

            # Master grad buffers on this rank
            fc1_grads = [
                torch.randn(args.expert_fc1_numel, device="cuda", dtype=torch.float32)
                for _ in range(args.num_local_master_experts)
            ]
            fc2_grads = [
                torch.randn(args.expert_fc2_numel, device="cuda", dtype=torch.float32)
                for _ in range(args.num_local_master_experts)
            ]

            # Dummy weights for initialization
            fc1_weights = [
                torch.empty(0, device="cuda", dtype=torch.bfloat16)
                for _ in range(args.num_local_master_experts)
            ]
            fc2_weights = [
                torch.empty(0, device="cuda", dtype=torch.bfloat16)
                for _ in range(args.num_local_master_experts)
            ]

            manager.construct_local_master_ptr_pool(
                layer_id, fc1_weights, fc2_weights, fc1_grads, fc2_grads
            )

            # Fill replica buffers on this rank with some values
            # These will be pulled by the master ranks of the logical experts they replicate.
            replica_grad_buffer = (
                manager.local_replica_grad_buffer
            )  # [num_local_redundant, expert_total_numel]
            replica_grad_buffer_ref = torch.randn(
                replica_grad_buffer.shape, device="cuda", dtype=torch.float32
            )
            replica_grad_buffer.copy_(replica_grad_buffer_ref)

            # Precalculate golden results using all-reduce
            global_logical_expert_fc1_grad_buffer = torch.zeros(
                (manager.num_global_logical_experts, args.expert_fc1_numel),
                device="cuda",
                dtype=torch.float32,
            )
            global_logical_expert_fc2_grad_buffer = torch.zeros(
                (manager.num_global_logical_experts, args.expert_fc2_numel),
                device="cuda",
                dtype=torch.float32,
            )
            for i in range(manager.num_local_physical_experts):
                global_phys_idx = rank * manager.num_local_physical_experts + i
                global_log_idx = manager.physical_to_logical_map[
                    layer_id, global_phys_idx
                ].item()
                if i < manager.num_local_master_experts:
                    global_logical_expert_fc1_grad_buffer[global_log_idx, :].copy_(
                        fc1_grads[i]
                    )
                    global_logical_expert_fc2_grad_buffer[global_log_idx, :].copy_(
                        fc2_grads[i]
                    )
                else:
                    local_replica_offset = i - manager.num_local_master_experts
                    global_logical_expert_fc1_grad_buffer[global_log_idx, :].copy_(
                        replica_grad_buffer[
                            local_replica_offset, : args.expert_fc1_numel
                        ]
                    )
                    global_logical_expert_fc2_grad_buffer[global_log_idx, :].copy_(
                        replica_grad_buffer[
                            local_replica_offset, args.expert_fc1_numel :
                        ]
                    )
            dist.all_reduce(global_logical_expert_fc1_grad_buffer, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_logical_expert_fc2_grad_buffer, op=dist.ReduceOp.SUM)

            # Verify correctness at first run
            replica_grad_buffer.copy_(replica_grad_buffer_ref)
            dist.barrier()
            manager.grad_reduce(layer_id, mode=mode, async_finish=False)
            dist.barrier()
            assert (
                (replica_grad_buffer == 0).all().item()
            ), f"Replica grad buffer was not zeroed out on rank {rank}"
            for i in range(manager.num_local_master_experts):
                global_phys_idx = rank * manager.num_local_physical_experts + i
                global_log_idx = manager.physical_to_logical_map[
                    layer_id, global_phys_idx
                ].item()
                assert torch.allclose(
                    fc1_grads[i],
                    global_logical_expert_fc1_grad_buffer[global_log_idx, :],
                    atol=args.correct_tolerance,
                ), f"FC1 grad mismatch for logical expert {global_log_idx} on rank {rank}"
                assert torch.allclose(
                    fc2_grads[i],
                    global_logical_expert_fc2_grad_buffer[global_log_idx, :],
                    atol=args.correct_tolerance,
                ), f"FC2 grad mismatch for logical expert {global_log_idx} on rank {rank}"
            torch.cuda.synchronize()
            dist.barrier()
            print_on_leader_ranks("*** Correctness verification passed! ***")

            # Performance benchmark
            grad_reduce_fn = lambda: manager.grad_reduce(
                layer_id, mode=mode, async_finish=False
            )
            pre_fn = lambda: replica_grad_buffer.copy_(replica_grad_buffer_ref)

            def grad_reduce_fn_full():
                replica_grad_buffer.copy_(replica_grad_buffer_ref)
                manager.grad_reduce(layer_id, mode=mode, async_finish=False)

            avg_time, min_time, max_time = bench(
                grad_reduce_fn,
                num_warmups=args.warmup_iters,
                num_tests=args.bench_iters,
                use_barrier=True,
                pre_fn=pre_fn,
            )
            avg_time_ms = avg_time * 1000
            min_time_ms = min_time * 1000
            max_time_ms = max_time * 1000

            kernel_names = (
                ("grad_reduce_kernel_low_sm",)
                if mode == "low_sm"
                else ("grad_reduce_kernel_high_sm",)
            )
            kernel_durations = bench_kineto(
                grad_reduce_fn_full,
                kernel_names=kernel_names,
                barrier_comm_profiling=True,
            )
            kernel_dur_second_this_rank = kernel_durations[0]
            kernel_dur_ms_this_rank = kernel_dur_second_this_rank * 1000

            bytes_recv_this_rank = 0
            for i in range(args.num_local_master_experts):
                global_phys_idx = rank * manager.num_local_physical_experts + i
                global_log_idx = manager.physical_to_logical_map[
                    layer_id, global_phys_idx
                ].item()
                num_replicas = (
                    manager.logical_replica_counts[layer_id, global_log_idx].item() - 1
                )
                bytes_recv_this_rank += (
                    num_replicas
                    * expert_total_numel
                    * replica_grad_buffer_ref.element_size()
                )
            data_recv_GB_this_rank = bytes_recv_this_rank / (1024**3)
            bandwidth_GBps_this_rank = (
                data_recv_GB_this_rank / kernel_dur_second_this_rank
                if kernel_dur_second_this_rank > 1e-6
                else 0
            )

            print(
                f"[Rank {rank}] kernel duration: {kernel_dur_ms_this_rank:.3f} ms, "
                f"data recv: {data_recv_GB_this_rank:.3f} GB, "
                f"bandwidth: {bandwidth_GBps_this_rank:.2f} GB/s",
                flush=True,
            )
            dist.barrier()

            # Calculate bandwidth
            avg_data_bytes = (
                manager.num_local_redundant_experts
                * expert_total_numel
                * replica_grad_buffer_ref.element_size()
            )
            avg_data_MB = avg_data_bytes / (1024**2)
            avg_data_GB = avg_data_bytes / (1024**3)
            avg_bandwidth_GBps = avg_data_GB / (avg_time_ms / 1000.0)

            print_on_leader_ranks(f"-" * 80)
            print_on_leader_ranks("Performance metrics:")
            print_on_leader_ranks(
                f"  - E2E Latency: {avg_time_ms:.3f} ms (avg) | {min_time_ms:.3f} ms (min) | {max_time_ms:.3f} ms (max)"
            )
            print_on_leader_ranks(
                f"  - Average Data Moved (per-rank): {avg_data_MB} MB"
            )
            print_on_leader_ranks(
                f"  - End2end Bandwidth: {avg_bandwidth_GBps:.2f} GB/s"
            )
            print_on_leader_ranks("\n")

    manager.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
