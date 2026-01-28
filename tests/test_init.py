import torch
import torch.distributed as dist
import os
import ultra_ep


def test_init():
    # Initialize distributed environment
    if not dist.is_initialized():
        # Expecting to be run with torchrun which sets MASTER_ADDR, MASTER_PORT, etc.
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Starting distributed initialization test with {world_size} ranks...")

    # Parameters for Manager:
    # group, num_local_master_experts, num_local_redundant_experts, expert_fc1_numel, expert_fc2_numel
    num_local_master_experts = 4
    num_local_redundant_experts = 2
    expert_fc1_numel = 2048 * 4096
    expert_fc2_numel = 1024 * 4096

    try:
        if rank == 0:
            print("Initializing ultra_ep.Manager...")
            print(f"  num_local_master_experts: {num_local_master_experts}")
            print(f"  num_local_redundant_experts: {num_local_redundant_experts}")
            print(f"  expert_fc1_numel: {expert_fc1_numel}")
            print(f"  expert_fc2_numel: {expert_fc2_numel}")

        # Manager.__init__ internally calls:
        # 1. init_runtime: which performs NVSHMEM initialization and unique ID synchronization.
        # 2. sync_ipc_handles: which performs IPC handle exchange across all ranks.
        manager = ultra_ep.Manager(
            group=dist.group.WORLD,
            num_local_master_experts=num_local_master_experts,
            num_local_redundant_experts=num_local_redundant_experts,
            expert_fc1_numel=expert_fc1_numel,
            expert_fc2_numel=expert_fc2_numel,
            explicitly_destroy=True,
        )

        # Verify Python-side properties
        assert manager.num_local_master_experts == num_local_master_experts
        assert manager.num_local_redundant_experts == num_local_redundant_experts
        assert manager.expert_fc1_numel == expert_fc1_numel
        assert manager.expert_fc2_numel == expert_fc2_numel

        # Check if the manager is successfully initialized and available
        # availability is set to true after sync_global_ipc_handles completes in C++.
        is_available = manager.runtime.is_available()
        print(f"Rank {rank}: Manager initialized. Available: {is_available}")

        assert (
            is_available
        ), f"Rank {rank}: Manager should be available after initialization"

        # Synchronization barrier to ensure all ranks reached this point
        dist.barrier()
        if rank == 0:
            print("All ranks successfully initialized and exchanged IPC handles.")

        # Test destruction
        if rank == 0:
            print("Testing manager destruction...")

        manager.destroy()

        if rank == 0:
            print("Manager destroyed successfully.")

    except Exception as e:
        print(
            f"Rank {rank}: Initialization or communication test failed with error: {e}"
        )
        # Ensure we don't hang other ranks if one fails
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e

    dist.barrier()
    if rank == 0:
        print("Test 'test_init' passed!")


if __name__ == "__main__":
    test_init()
