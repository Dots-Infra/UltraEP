import os
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional

import ultra_ep._C as _C
from .runtime import init_runtime, sync_ipc_handles


class Manager:
    def __init__(
        self,
        group: dist.ProcessGroup,
        num_local_master_experts: int,
        num_local_redundant_experts: int,
        expert_fc1_numel: int,
        expert_fc2_numel: int,
        explicitly_destroy: bool = False,
    ):
        # Initialize global nvshmem runtime (if not initialized)
        init_runtime(group)

        self.group = group
        self.device = torch.cuda.current_device()
        self.rank = group.rank()
        self.num_ranks = group.size()

        self.num_local_master_experts = num_local_master_experts
        self.num_local_redundant_experts = num_local_redundant_experts
        self.expert_fc1_numel = expert_fc1_numel
        self.expert_fc2_numel = expert_fc2_numel
        self.expert_total_numel = expert_fc1_numel + expert_fc2_numel
        self.num_local_physical_experts = (
            num_local_master_experts + num_local_redundant_experts
        )
        self.num_global_physical_experts = (
            self.num_local_physical_experts * self.num_ranks
        )
        self.num_global_logical_experts = num_local_master_experts * self.num_ranks

        # Create cpp handle
        self.explicitly_destroy = explicitly_destroy
        self.runtime = _C.Manager(
            num_local_master_experts,
            num_local_redundant_experts,
            expert_fc1_numel,
            expert_fc2_numel,
            explicitly_destroy,
        )
        sync_ipc_handles(self.runtime)
        assert self.runtime.is_available()

        # Placement on CPU
        self.physical_to_logical_map: torch.Tensor = (
            self.runtime.get_physical_to_logical_map_tensor()
        )
        self.logical_to_physical_map: torch.Tensor = (
            self.runtime.get_logical_to_physical_map_tensor()
        )
        self.logical_replica_counts: torch.Tensor = (
            self.runtime.get_logical_replica_counts_tensor()
        )
        # Replica weight/grad buffers shared by layers on GPU
        # Blobbed from cpp-side buffers
        self.local_replica_weight_buffer: torch.Tensor = (
            self.runtime.get_local_replica_weight_buffer_tensor()
        )
        self.local_replica_grad_buffer: torch.Tensor = (
            self.runtime.get_local_replica_grad_buffer_tensor()
        )

        self.check_tensors_blob_from_cpp()

    def destroy(self):
        assert self.explicitly_destroy

        if self.runtime is not None:
            self.runtime.destroy()
        self.runtime = None

    def check_tensors_blob_from_cpp(self):
        assert (
            self.physical_to_logical_map.device == torch.device("cpu")
            and self.logical_to_physical_map.device == torch.device("cpu")
            and self.logical_replica_counts.device == torch.device("cpu")
        )
        assert (
            self.physical_to_logical_map.dtype == torch.int32
            and self.logical_to_physical_map.dtype == torch.int32
            and self.logical_replica_counts.dtype == torch.int32
        )
        assert (
            self.physical_to_logical_map.shape == (self.num_global_physical_experts,)
            and self.logical_to_physical_map.shape
            == (self.num_global_logical_experts, self.num_ranks)
            and self.logical_replica_counts.shape == (self.num_global_logical_experts,)
        )
        assert (self.physical_to_logical_map == -1).all().item()
        assert (self.logical_to_physical_map == -1).all().item()
        assert self.local_replica_weight_buffer.device == torch.device(
            "cuda", self.device
        )
        assert self.local_replica_grad_buffer.device == torch.device(
            "cuda", self.device
        )
        assert self.local_replica_weight_buffer.dtype == torch.bfloat16
        assert self.local_replica_grad_buffer.dtype == torch.float32
        assert self.local_replica_weight_buffer.shape == (
            self.num_local_redundant_experts,
            self.expert_total_numel,
        )
        assert self.local_replica_grad_buffer.shape == (
            self.num_local_redundant_experts,
            self.expert_total_numel,
        )
