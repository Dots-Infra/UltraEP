import torch
import torch.distributed as dist
from typing import List

import ultra_ep._C as _C
from .runtime import init_runtime

MAX_MODEL_LAYERS = 200


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
        self.nvl_domain_size = init_runtime(group)

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

        # Pre-configured during model construction
        # Shape: num_layers x [num_local_master_experts,]
        self.local_master_fc1_weight_pool = [None] * MAX_MODEL_LAYERS
        self.local_master_fc2_weight_pool = [None] * MAX_MODEL_LAYERS
        self.local_master_fc1_grad_pool = [None] * MAX_MODEL_LAYERS
        self.local_master_fc2_grad_pool = [None] * MAX_MODEL_LAYERS

        # Create cpp handle
        self.explicitly_destroy = explicitly_destroy
        self.runtime = _C.Manager(
            num_local_master_experts,
            num_local_redundant_experts,
            expert_fc1_numel,
            expert_fc2_numel,
            explicitly_destroy,
        )
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

    def construct_local_master_ptr_pool(
        self,
        layer_id: int,
        fc1_weights: List[torch.Tensor],
        fc2_weights: List[torch.Tensor],
        fc1_grads: List[torch.Tensor],
        fc2_grads: List[torch.Tensor],
    ):
        assert layer_id < MAX_MODEL_LAYERS
        assert len(fc1_weights) == self.num_local_master_experts
        assert len(fc2_weights) == self.num_local_master_experts
        assert len(fc1_grads) == self.num_local_master_experts
        assert len(fc2_grads) == self.num_local_master_experts

        def _to_dataptr_tensor(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [t.data_ptr() for t in tensors], dtype=torch.int64, device="cpu"
            ).contiguous()

        self.local_master_fc1_weight_pool[layer_id] = _to_dataptr_tensor(fc1_weights)
        self.local_master_fc2_weight_pool[layer_id] = _to_dataptr_tensor(fc2_weights)
        self.local_master_fc1_grad_pool[layer_id] = _to_dataptr_tensor(fc1_grads)
        self.local_master_fc2_grad_pool[layer_id] = _to_dataptr_tensor(fc2_grads)

    def grad_reduce(self, layer_id: int):
        assert layer_id < MAX_MODEL_LAYERS
        assert (
            self.local_master_fc1_grad_pool[layer_id] is not None
            and self.local_master_fc2_grad_pool[layer_id] is not None
        )
        self.runtime.grad_reduce(
            self.local_master_fc1_grad_pool[layer_id],
            self.local_master_fc2_grad_pool[layer_id],
        )

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
