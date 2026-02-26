import os
import torch
import torch.distributed as dist
from typing import List, Optional
from datetime import datetime

import ultra_ep._C as _C
from .runtime import init_runtime
from .event import EventHandle
from .reroute import _RerouteProbsFunction, _RerouteCUDAFunction
from .util import get_max_by_mean


class Manager:
    def __init__(
        self,
        group: dist.ProcessGroup,
        num_layers: int,
        num_local_master_experts: int,
        num_local_redundant_experts: int,
        expert_fc1_numel: int,
        expert_fc2_numel: int,
        is_train: bool = True,
        explicitly_destroy: bool = False,
        log_expert_loads: bool = False,
    ):
        # Initialize global nvshmem runtime (if not initialized)
        self.nvl_domain_size = init_runtime(group)

        self.group = group
        self.device = torch.cuda.current_device()
        self.rank = group.rank()
        self.num_ranks = group.size()

        self.num_layers = num_layers
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
        self.is_train = is_train

        # Pre-configured during model construction
        # Shape: num_layers x [num_local_master_experts,]
        self.num_alloc_layers = (
            num_layers + 3
        )  # In case layer_id begins from 1 instead of 0
        self.local_master_fc1_weight_pool = [None] * self.num_alloc_layers
        self.local_master_fc2_weight_pool = [None] * self.num_alloc_layers
        self.local_master_fc1_grad_pool = [None] * self.num_alloc_layers
        self.local_master_fc2_grad_pool = [None] * self.num_alloc_layers

        # Expert loads logging
        self.log_expert_loads = log_expert_loads
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ep_loads_save_dir = os.environ.get(
            "EP_LOADS_SAVE_DIR", "/var/log/ep_loads"
        )
        os.makedirs(self.ep_loads_save_dir, exist_ok=True)
        self.ep_log_path = os.path.join(
            self.ep_loads_save_dir, f"ep_loads_{now_str}.log"
        )

        # Create cpp handle
        self.explicitly_destroy = explicitly_destroy
        self.runtime = _C.Manager(
            self.num_alloc_layers,
            num_local_master_experts,
            num_local_redundant_experts,
            expert_fc1_numel,
            expert_fc2_numel,
            is_train,
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
        assert layer_id < self.num_alloc_layers
        assert len(fc1_weights) == self.num_local_master_experts
        assert len(fc2_weights) == self.num_local_master_experts
        assert len(fc1_grads) == self.num_local_master_experts
        assert len(fc2_grads) == self.num_local_master_experts

        def check_tensors_dtype(tensors: List[torch.Tensor], dtype: torch.dtype):
            for t in tensors:
                assert (
                    t.dtype == dtype
                ), f"Expected weight/grad dtype {dtype}, got {t.dtype}"

        check_tensors_dtype(fc1_weights, torch.bfloat16)
        check_tensors_dtype(fc2_weights, torch.bfloat16)
        check_tensors_dtype(fc1_grads, torch.float32)
        check_tensors_dtype(fc2_grads, torch.float32)

        def _to_dataptr_tensor(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [t.data_ptr() for t in tensors], dtype=torch.int64, device="cpu"
            ).contiguous()

        self.local_master_fc1_weight_pool[layer_id] = _to_dataptr_tensor(fc1_weights)
        self.local_master_fc2_weight_pool[layer_id] = _to_dataptr_tensor(fc2_weights)
        self.local_master_fc1_grad_pool[layer_id] = _to_dataptr_tensor(fc1_grads)
        self.local_master_fc2_grad_pool[layer_id] = _to_dataptr_tensor(fc2_grads)

    def grad_reduce(
        self,
        layer_id: int,
        mode: str = "low_sm",
        previous_event: Optional[EventHandle] = None,
        async_finish: bool = False,
    ):
        assert layer_id < self.num_alloc_layers
        assert (
            self.local_master_fc1_grad_pool[layer_id] is not None
            and self.local_master_fc2_grad_pool[layer_id] is not None
        )
        event = self.runtime.grad_reduce(
            layer_id,
            self.local_master_fc1_grad_pool[layer_id],
            self.local_master_fc2_grad_pool[layer_id],
            mode,
            getattr(previous_event, "event", None),
            async_finish,
        )
        return EventHandle(event)

    def weight_sync(
        self,
        layer_id: int,
        previous_event: Optional[EventHandle] = None,
        async_finish: bool = False,
    ):
        """
        Synchronize master weights to replicas.

        Each local master broadcasts its weight to all corresponding remote replicas.
        This is optimized for hot masters with multiple replicas - the weight is loaded
        to shared memory once and then TMA stored to all replica destinations.

        Args:
            layer_id: Layer index for which to sync weights
            previous_event: Optional event to wait for before starting
            async_finish: If True, return immediately with an event handle
                         If False, wait for completion before returning

        Returns:
            EventHandle handle if async_finish=True, else None
        """
        assert layer_id < self.num_alloc_layers
        assert (
            self.local_master_fc1_weight_pool[layer_id] is not None
            and self.local_master_fc2_weight_pool[layer_id] is not None
        )
        with torch.cuda.nvtx.range(f"Launch weight_sync (layer {layer_id})"):
            event = self.runtime.weight_sync(
                layer_id,
                self.local_master_fc1_weight_pool[layer_id],
                self.local_master_fc2_weight_pool[layer_id],
                getattr(previous_event, "event", None),
                async_finish,
            )
            return EventHandle(event)

    def update_placement(
        self,
        layer_id: int,
        routing_map: torch.Tensor,
        verify_reduced_loads: bool = False,
    ):
        """
        Update expert placement for a single layer based on real-time load statistics.

        Runs the EPLB-style placement algorithm entirely on CPU:
          1. Masters remain fixed at their pre-assigned positions.
          2. Per NVL domain, greedily replicates the most loaded experts.
          3. Per NVL domain, packs replicas to GPU slots via LPT bin-packing,
             ensuring replicas are never placed on the same GPU as their master.

        Deterministic: all ranks compute identical results, no broadcast needed.

        Args:
            layer_id: The MoE layer index to update.
            routing_map: [num_tokens, num_global_logical_experts] bool tensor, logical routing map.
        """
        assert layer_id < self.num_alloc_layers
        with torch.cuda.nvtx.range(f"Update placement (layer {layer_id})"):
            self.runtime.update_placement(layer_id, routing_map)
        if verify_reduced_loads:
            global_logical_expert_loads = routing_map.sum(dim=0, dtype=torch.int32)
            dist.all_reduce(global_logical_expert_loads, group=self.group)
            assert torch.equal(
                global_logical_expert_loads,
                self.runtime.get_global_logical_expert_loads_tensor(),
            )

    def reroute(
        self,
        layer_id: int,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        backend: str = "cuda",
    ):
        """
        Expand routing from logical experts to physical experts using deterministic
        round-robin dispatch.

        For each logical expert l with C_l = lcnts[l] physical instances,
        the k-th token (by global token index) routed to l is dispatched to
        physical expert l2p[l, k % C_l].

        Two paths are available (selected by backend):
          - CPU path: C++ RerouteSolver computes index arrays, Python scatters.
          - CUDA path: fused GPU kernel avoids all H2D/D2H transfers.

        Args:
            layer_id: The MoE layer index to reroute.
            probs: [num_tokens, num_logical_experts] float tensor, routing probabilities.
            routing_map: [num_tokens, num_logical_experts] bool tensor, logical routing map.
        Returns:
            expanded_probs: [num_tokens, num_physical_experts] float tensor, expanded probabilities.
            expanded_routing_map: [num_tokens, num_physical_experts] bool tensor, physical routing map.
        """
        if self.log_expert_loads:
            orig_log_expert_loads, balanced_phys_expert_loads = None, None
            orig_log_expert_loads = routing_map.sum(dim=0, dtype=torch.int32)
            dist.all_reduce(orig_log_expert_loads, group=self.group)
            curr_phy2log = self.physical_to_logical_map[layer_id].clone().cpu()
            max_replica_cnt = self.logical_replica_counts[layer_id].cpu().max().item()

        if backend == "cuda":
            expanded_probs, expanded_routing_map = self._reroute_cuda(
                layer_id, probs, routing_map
            )
        elif backend == "cpu":
            expanded_probs, expanded_routing_map = self._reroute_cpu(
                layer_id, probs, routing_map
            )
        else:
            raise ValueError(f"Invalid backend: {backend}")

        if self.log_expert_loads:
            balanced_phys_expert_loads = expanded_routing_map.sum(
                dim=0, dtype=torch.int32
            )
            dist.all_reduce(balanced_phys_expert_loads, group=self.group)

            if self.rank == 0:
                mask = curr_phy2log != -1  # filter empty slots for physical experts
                balanced_phys_expert_loads = balanced_phys_expert_loads.cpu()[mask]

                orig_imbalance = get_max_by_mean(orig_log_expert_loads.cpu())
                balanced_imbalance = get_max_by_mean(balanced_phys_expert_loads)
                with open(self.ep_log_path, "a") as f:
                    f.write(
                        f"Layer {layer_id}: Imbalance (max/mean load): {orig_imbalance:.2f} -> "
                        f"{balanced_imbalance:.2f} ({orig_imbalance / balanced_imbalance:.2f}x) | "
                        f"max #replicas: {max_replica_cnt}\n"
                    )

        return expanded_probs, expanded_routing_map

    def _reroute_cpu(
        self,
        layer_id: int,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ):
        """CPU path: C++ RerouteSolver + Python index scatter/gather."""
        num_tokens = routing_map.size(0)
        num_global_physical = self.num_global_physical_experts
        token_indices, logical_indices, physical_indices = self.runtime.reroute_cpu(
            layer_id, routing_map
        )

        # Construct expanded_routing_map on the same device as routing_map
        expanded_routing_map = torch.zeros(
            num_tokens, num_global_physical, dtype=torch.bool, device=routing_map.device
        )
        if token_indices.numel() > 0:
            expanded_routing_map[token_indices, physical_indices] = True

        # Construct expanded_probs with appropriate gradient handling
        if self.is_train and probs.requires_grad:
            expanded_probs = _RerouteProbsFunction.apply(
                probs,
                token_indices,
                logical_indices,
                physical_indices,
                num_global_physical,
            )
        else:
            # Inference path: no autograd overhead
            expanded_probs = torch.zeros(
                num_tokens,
                num_global_physical,
                dtype=probs.dtype,
                device=routing_map.device,
            )
            if token_indices.numel() > 0:
                expanded_probs[token_indices, physical_indices] = probs[
                    token_indices, logical_indices
                ]

        return expanded_probs, expanded_routing_map

    def _reroute_cuda(
        self,
        layer_id: int,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ):
        """CUDA path: fused GPU kernel for reroute with pre-allocated buffers."""
        if self.is_train and probs.requires_grad:
            expanded_probs, expanded_routing_map = _RerouteCUDAFunction.apply(
                probs,
                routing_map,
                self.runtime,
                layer_id,
            )
        else:
            # Inference path: direct Manager call, no autograd overhead
            with torch.cuda.nvtx.range(f"Reroute CUDA forward (layer {layer_id})"):
                expanded_probs, expanded_routing_map = (
                    self.runtime.reroute_cuda_forward(layer_id, probs, routing_map)
                )

        return expanded_probs, expanded_routing_map

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
            self.physical_to_logical_map.shape
            == (
                self.num_alloc_layers,
                self.num_global_physical_experts,
            )
            and self.logical_to_physical_map.shape
            == (self.num_alloc_layers, self.num_global_logical_experts, self.num_ranks)
            and self.logical_replica_counts.shape
            == (
                self.num_alloc_layers,
                self.num_global_logical_experts,
            )
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

    def get_comm_stream(self) -> torch.Stream:
        ts: torch.Stream = self.runtime.get_comm_stream()
        return torch.cuda.Stream(
            stream_id=ts.stream_id,
            device_index=ts.device_index,
            device_type=ts.device_type,
        )
