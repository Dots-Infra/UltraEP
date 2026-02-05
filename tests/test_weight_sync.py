import torch
import torch.distributed as dist
import os, sys
import argparse
import time
import ultra_ep
from ultra_ep.util import print_rank_0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import setup_placement, bench, bench_kineto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-local-master-experts", type=int, default=4)
    parser.add_argument("--num-local-redundant-experts", type=int, default=2)
    parser.add_argument("--expert-fc1-numel", type=int, default=3072 * 4096)
    parser.add_argument("--expert-fc2-numel", type=int, default=1536 * 4096)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--bench-iters", type=int, default=100)
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
        num_local_master_experts=args.num_local_master_experts,
        num_local_redundant_experts=args.num_local_redundant_experts,
        expert_fc1_numel=args.expert_fc1_numel,
        expert_fc2_numel=args.expert_fc2_numel,
        explicitly_destroy=True,
    )

    num_nvl_ranks = manager.nvl_domain_size
    print_rank_0(
        f"Running weight_sync test with {world_size} ranks, {num_nvl_ranks} NVL ranks"
    )
    print_rank_0(
        f"Local experts: {args.num_local_master_experts} master, {args.num_local_redundant_experts} redundant"
    )
    print_rank_0(f"Numel: FC1={args.expert_fc1_numel}, FC2={args.expert_fc2_numel}")

    for replica_distrib in ["uniform", "skewed"]:
        print_rank_0(f"=" * 80)
        print_rank_0(f"Test weight_sync with {replica_distrib} distribution")
        print_rank_0(f"=" * 80)

        setup_placement(
            world_size,
            args.num_local_master_experts,
            args.num_local_redundant_experts,
            manager.physical_to_logical_map,
            manager.logical_to_physical_map,
            manager.logical_replica_counts,
            replica_distribution=replica_distrib,
            num_nvl_ranks=num_nvl_ranks,
            hot_expert_ratio_per_nvl_domain=0.03,
            seed=args.seed,
        )

        # Count replicas for debugging
        total_replicas = 0
        max_replicas = 0
        for i in range(args.num_local_master_experts):
            global_phys_idx = rank * manager.num_local_physical_experts + i
            global_log_idx = manager.physical_to_logical_map[global_phys_idx].item()
            num_replicas = manager.logical_replica_counts[global_log_idx].item() - 1
            total_replicas += num_replicas
            max_replicas = max(max_replicas, num_replicas)

        # All-reduce to get global stats
        total_replicas_tensor = torch.tensor([total_replicas], device="cuda")
        max_replicas_tensor = torch.tensor([max_replicas], device="cuda")
        dist.all_reduce(total_replicas_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_replicas_tensor, op=dist.ReduceOp.MAX)
        print_rank_0(
            f"Total replica tasks across all ranks: {total_replicas_tensor.item()}"
        )
        print_rank_0(f"Max replicas per master: {max_replicas_tensor.item()}")

        # Prepare data
        layer_id = 0
        expert_total_numel = args.expert_fc1_numel + args.expert_fc2_numel

        # Master weight buffers on this rank (source)
        fc1_weights = [
            torch.randn(args.expert_fc1_numel, device="cuda", dtype=torch.bfloat16)
            for _ in range(args.num_local_master_experts)
        ]
        fc2_weights = [
            torch.randn(args.expert_fc2_numel, device="cuda", dtype=torch.bfloat16)
            for _ in range(args.num_local_master_experts)
        ]

        # Dummy grads for initialization
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

        # Randomly init replica weight buffer, as they are reused across layers
        replica_weight_buffer = manager.local_replica_weight_buffer
        replica_weight_buffer.random_(0, 3)

        # Build reference: what each replica should contain after sync
        # For each local replica, find its master and get the expected weight
        expected_replica_weights = torch.zeros_like(replica_weight_buffer)

        for i in range(args.num_local_redundant_experts):
            local_phys_idx = args.num_local_master_experts + i
            global_phys_idx = rank * manager.num_local_physical_experts + local_phys_idx
            logical_idx = manager.physical_to_logical_map[global_phys_idx].item()

            if logical_idx < 0:
                continue  # Not assigned

            # Find the master for this logical expert
            master_global_phys_idx = manager.logical_to_physical_map[
                logical_idx, 0
            ].item()
            master_rank = master_global_phys_idx // manager.num_local_physical_experts
            master_local_idx = (
                master_global_phys_idx % manager.num_local_physical_experts
            )

            # The master's weight needs to be broadcast to build reference
            # We'll use all_gather to get all master weights

        # Build global master weight buffer via all_gather
        local_master_fc1_weights = torch.stack(
            fc1_weights
        )  # [num_local_master, fc1_numel]
        local_master_fc2_weights = torch.stack(
            fc2_weights
        )  # [num_local_master, fc2_numel]

        global_master_fc1_weights_list = [
            torch.zeros_like(local_master_fc1_weights) for _ in range(world_size)
        ]
        global_master_fc2_weights_list = [
            torch.zeros_like(local_master_fc2_weights) for _ in range(world_size)
        ]

        dist.all_gather(global_master_fc1_weights_list, local_master_fc1_weights)
        dist.all_gather(global_master_fc2_weights_list, local_master_fc2_weights)

        # [world_size, num_local_master, numel]
        global_master_fc1_weights = torch.stack(global_master_fc1_weights_list)
        global_master_fc2_weights = torch.stack(global_master_fc2_weights_list)

        # Build expected replica weights
        for i in range(args.num_local_redundant_experts):
            local_phys_idx = args.num_local_master_experts + i
            global_phys_idx = rank * manager.num_local_physical_experts + local_phys_idx
            logical_idx = manager.physical_to_logical_map[global_phys_idx].item()

            if logical_idx < 0:
                continue

            # Find the master
            master_global_phys_idx = manager.logical_to_physical_map[
                logical_idx, 0
            ].item()
            master_rank = master_global_phys_idx // manager.num_local_physical_experts
            master_local_idx = (
                master_global_phys_idx % manager.num_local_physical_experts
            )

            # Copy expected values
            expected_replica_weights[i, : args.expert_fc1_numel] = (
                global_master_fc1_weights[master_rank, master_local_idx, :]
            )
            expected_replica_weights[i, args.expert_fc1_numel :] = (
                global_master_fc2_weights[master_rank, master_local_idx, :]
            )

        # Run weight_sync
        dist.barrier()
        manager.weight_sync(layer_id, async_finish=False)
        dist.barrier()

        # Verify correctness
        match = torch.allclose(
            replica_weight_buffer, expected_replica_weights, atol=args.correct_tolerance
        )

        if not match:
            # Find first mismatch for debugging
            diff = (replica_weight_buffer - expected_replica_weights).abs()
            max_diff = diff.max().item()
            max_diff_idx = diff.argmax().item()
            print(
                f"Rank {rank}: Weight sync mismatch! Max diff: {max_diff} at index {max_diff_idx}",
                flush=True,
            )
            # Check each replica
            for i in range(args.num_local_redundant_experts):
                replica_diff = (
                    (replica_weight_buffer[i] - expected_replica_weights[i])
                    .abs()
                    .max()
                    .item()
                )
                if replica_diff > args.correct_tolerance:
                    local_phys_idx = args.num_local_master_experts + i
                    global_phys_idx = (
                        rank * manager.num_local_physical_experts + local_phys_idx
                    )
                    logical_idx = manager.physical_to_logical_map[
                        global_phys_idx
                    ].item()
                    print(
                        f"  Replica {i} (logical {logical_idx}): max diff = {replica_diff}",
                        flush=True,
                    )

        assert match, f"Weight sync verification failed on rank {rank}"

        torch.cuda.synchronize()
        dist.barrier()
        print_rank_0("*** Correctness verification passed! ***")

        # Performance benchmark
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
        # Calculate bandwidth
        # For weight sync, we send data from masters to replicas
        # Each master sends to all its replicas
        bytes_sent_this_rank = 0
        for i in range(args.num_local_master_experts):
            global_phys_idx = rank * manager.num_local_physical_experts + i
            global_log_idx = manager.physical_to_logical_map[global_phys_idx].item()
            num_replicas = manager.logical_replica_counts[global_log_idx].item() - 1
            bytes_sent_this_rank += (
                num_replicas * expert_total_numel * 2
            )  # bf16 = 2 bytes
        data_sent_GB_this_rank = bytes_sent_this_rank / (1024**3)
        bandwidth_GBps_this_rank = (
            data_sent_GB_this_rank / kernel_dur_second_this_rank
            if kernel_dur_second_this_rank > 1e-6
            else 0
        )

        print(
            f"[Rank {rank}] kernel duration: {kernel_dur_ms_this_rank:.3f} ms, data sent: {data_sent_GB_this_rank:.3f} GB, bandwidth: {bandwidth_GBps_this_rank:.2f} GB/s",
            flush=True,
        )
        dist.barrier()

        avg_data_bytes = (
            manager.num_local_redundant_experts
            * expert_total_numel
            * 2  # bf16 = 2 bytes
        )
        avg_data_GB = avg_data_bytes / (1024**3)
        avg_bandwidth_GBps = avg_data_GB / (avg_time_ms / 1000.0)

        print_rank_0(f"-" * 80)
        print_rank_0("Performance metrics:")
        print_rank_0(
            f"  - E2E Latency: {avg_time_ms:.3f} ms (avg) | {min_time_ms:.3f} ms (min) | {max_time_ms:.3f} ms (max)"
        )
        print_rank_0(f"  - Average Data Sent (all ranks): {avg_data_GB:.3f} GB")
        print_rank_0(
            f"  - Average Bandwidth (all ranks): {avg_bandwidth_GBps:.2f} GB/s\n"
        )

    manager.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
