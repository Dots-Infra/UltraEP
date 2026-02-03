import torch
import torch.distributed as dist
import os, sys
import argparse
import time
import ultra_ep
from ultra_ep.util import print_rank_0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import setup_placement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-local-master-experts", type=int, default=4)
    parser.add_argument("--num-local-redundant-experts", type=int, default=2)
    parser.add_argument("--expert-fc1-numel", type=int, default=3072 * 4096)
    parser.add_argument("--expert-fc2-numel", type=int, default=1536 * 4096)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)
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
    print_rank_0(f"Running test with {world_size} ranks, {num_nvl_ranks} NVL ranks")
    print_rank_0(
        f"Local experts: {args.num_local_master_experts} master, {args.num_local_redundant_experts} redundant"
    )
    print_rank_0(f"Numel: FC1={args.expert_fc1_numel}, FC2={args.expert_fc2_numel}")

    for replica_distrib in ["uniform", "skewed"]:
        print_rank_0(f"=" * 80)
        print_rank_0(f"Test expert placement with {replica_distrib} distribution")
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

        # Prepare data
        layer_id = 0
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
            global_log_idx = manager.physical_to_logical_map[global_phys_idx].item()
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
                    replica_grad_buffer[local_replica_offset, : args.expert_fc1_numel]
                )
                global_logical_expert_fc2_grad_buffer[global_log_idx, :].copy_(
                    replica_grad_buffer[local_replica_offset, args.expert_fc1_numel :]
                )
        dist.all_reduce(global_logical_expert_fc1_grad_buffer, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_logical_expert_fc2_grad_buffer, op=dist.ReduceOp.SUM)

        # Verify correctness at first run
        replica_grad_buffer.copy_(replica_grad_buffer_ref)
        dist.barrier()
        manager.grad_reduce(layer_id)
        dist.barrier()
        assert (
            (replica_grad_buffer == 0).all().item()
        ), f"Replica grad buffer was not zeroed out on rank {rank}"
        for i in range(manager.num_local_master_experts):
            global_phys_idx = rank * manager.num_local_physical_experts + i
            global_log_idx = manager.physical_to_logical_map[global_phys_idx].item()
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
        dist.barrier()
        print_rank_0("*** Correctness verification passed! ***")

        # Warmup
        for _ in range(args.warmup_iters):
            # We need to refill replica buffers because grad_reduce zeros them out
            replica_grad_buffer.copy_(replica_grad_buffer_ref)
            manager.grad_reduce(layer_id)
            torch.cuda.synchronize()

        # Performance benchmark
        total_latency = 0
        for _ in range(args.bench_iters):
            replica_grad_buffer.copy_(replica_grad_buffer_ref)
            dist.barrier()
            start = time.perf_counter()
            manager.grad_reduce(layer_id)
            torch.cuda.synchronize()
            dist.barrier()
            end = time.perf_counter()
            total_latency += end - start
            assert (
                (replica_grad_buffer == 0).all().item()
            ), f"Replica grad buffer was not zeroed out on rank {rank}"

        avg_latency_ms = (total_latency / args.bench_iters) * 1000

        # Calculate bandwidth
        total_data_bytes = (
            world_size
            * manager.num_local_redundant_experts
            * expert_total_numel
            * replica_grad_buffer_ref.element_size()
        )
        total_data_MB = total_data_bytes / (1024**2)
        total_data_GB = total_data_bytes / (1024**3)
        avg_bandwidth_GBps = total_data_GB / (avg_latency_ms / 1000.0)

        print_rank_0("Performance metrics:")
        print_rank_0(f"  - Average Latency: {avg_latency_ms:.3f} ms")
        print_rank_0(f"  - Total Data Moved: {total_data_GB:.2f} GB")
        print_rank_0(f"  - End2end Bandwidth: {avg_bandwidth_GBps:.2f} GB/s")

    manager.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
