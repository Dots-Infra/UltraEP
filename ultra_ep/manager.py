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
        max_microbatches: int = 1,
        use_gpu_solver: bool = False,
        balance_threshold: float = 1.0,
        use_quota_solver: bool = False,
        quota_locality_aware: bool = True,
        quota_min_tokens_per_replica: int = 1,
        quota_allow_zero_master_quota: bool = True,
        quota_solver_version: int = 1,
        quota_v1_oracle_mode: int = 0,
        quota_v1_oracle_eps: float = 0.01,
        quota_v1_oracle_batch_k: int = 4,
        quota_v1_kernel_stage: int = 0,
    ):
        # Initialize global nvshmem runtime (if not initialized)
        self.nvl_domain_size = init_runtime(group)

        self.group = group
        self.id = id(group)
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

        # PP/VPP support: multiple micro-batches may be in-flight simultaneously.
        # Each (real_layer, microbatch_slot) pair gets a unique "virtual layer ID"
        # so that placement maps and reroute buffers don't collide across micro-batches.
        # For DDP-only (max_microbatches=1) the virtual ID equals the real layer ID.
        self.max_microbatches = max(1, max_microbatches)
        self.real_num_alloc_layers = num_layers + 3  # padding for 1-indexed layer IDs
        self.num_alloc_layers = self.real_num_alloc_layers * self.max_microbatches

        # Master weight/grad pointer pools, indexed by REAL layer_id.
        # These don't depend on micro-batch because the physical memory is the same.
        self.local_master_fc1_weight_pool = [None] * self.real_num_alloc_layers
        self.local_master_fc2_weight_pool = [None] * self.real_num_alloc_layers
        self.local_master_fc1_weight_pool_gpu = [None] * self.real_num_alloc_layers
        self.local_master_fc2_weight_pool_gpu = [None] * self.real_num_alloc_layers
        self.local_master_fc1_grad_pool = [None] * self.real_num_alloc_layers
        self.local_master_fc2_grad_pool = [None] * self.real_num_alloc_layers
        self.local_master_fc1_grad_pool_gpu = [None] * self.real_num_alloc_layers
        self.local_master_fc2_grad_pool_gpu = [None] * self.real_num_alloc_layers

        # Per-real-layer micro-batch slot counters (wraps modulo max_microbatches)
        self._mb_counters = [0] * self.real_num_alloc_layers

        # Expert loads logging
        self.log_expert_loads = os.environ.get("ULTRA_EP_LOG_EXPERT_LOADS", "0") == "1"
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ep_loads_save_dir = os.environ.get(
            "ULTRA_EP_LOADS_SAVE_DIR", "/var/log/ultra_ep_loads"
        )
        os.makedirs(self.ep_loads_save_dir, exist_ok=True)
        self.ep_log_path = os.path.join(
            self.ep_loads_save_dir, f"ep_loads_{now_str}_group{self.id}.log"
        )

        # Create cpp handle
        self.explicitly_destroy = explicitly_destroy
        self.use_gpu_solver = use_gpu_solver
        self.use_quota_solver = use_quota_solver
        # Quota solver keeps placement/reroute data on GPU, so grad/weight task build
        # should also use the GPU path to avoid hot-path D2H placement sync.
        self.use_gpu_task_build = self.use_gpu_solver or self.use_quota_solver
        self.runtime = _C.Manager(
            self.num_alloc_layers,
            num_local_master_experts,
            num_local_redundant_experts,
            expert_fc1_numel,
            expert_fc2_numel,
            is_train,
            explicitly_destroy,
            use_gpu_solver,
            balance_threshold,
            use_quota_solver,
            quota_locality_aware,
            quota_min_tokens_per_replica,
            quota_allow_zero_master_quota,
            quota_solver_version,
            quota_v1_oracle_mode,
            quota_v1_oracle_eps,
            quota_v1_oracle_batch_k,
            quota_v1_kernel_stage,
        )
        assert self.runtime.is_available()

        # Host-visible placement mirror. When use_gpu_solver=True the authoritative
        # copy lives on GPU and this mirror is refreshed on demand.
        self._physical_to_logical_map: torch.Tensor = (
            self.runtime.get_physical_to_logical_map_tensor()
        )
        self._logical_to_physical_map: torch.Tensor = (
            self.runtime.get_logical_to_physical_map_tensor()
        )
        self._logical_replica_counts: torch.Tensor = (
            self.runtime.get_logical_replica_counts_tensor()
        )
        self._logical_instance_quota: torch.Tensor = (
            self.runtime.get_logical_instance_quota_tensor()
        )
        self._logical_instance_quota_prefix: torch.Tensor = (
            self.runtime.get_logical_instance_quota_prefix_tensor()
        )
        self._rank_quota_prefix: torch.Tensor = (
            self.runtime.get_rank_quota_prefix_tensor()
        )
        # Replica weight buffer shared by layers on GPU
        self.local_replica_weight_buffer: torch.Tensor = (
            self.runtime.get_local_replica_weight_buffer_tensor()
        )
        # Grad buffer only available in training mode
        if self.is_train:
            self.local_replica_grad_buffer: torch.Tensor = (
                self.runtime.get_local_replica_grad_buffer_tensor()
            )
        else:
            self.local_replica_grad_buffer = None

        self.check_tensors_blob_from_cpp()

    def sync_placement_to_cpu(self, layer_id: Optional[int] = None):
        if layer_id is not None:
            assert layer_id < self.num_alloc_layers
        self.runtime.sync_placement_to_cpu(-1 if layer_id is None else layer_id)

    @property
    def physical_to_logical_map(self) -> torch.Tensor:
        self.sync_placement_to_cpu()
        return self._physical_to_logical_map

    @property
    def logical_to_physical_map(self) -> torch.Tensor:
        self.sync_placement_to_cpu()
        return self._logical_to_physical_map

    @property
    def logical_replica_counts(self) -> torch.Tensor:
        self.sync_placement_to_cpu()
        return self._logical_replica_counts

    @property
    def logical_instance_quota(self) -> torch.Tensor:
        self.sync_placement_to_cpu()
        return self._logical_instance_quota

    @property
    def logical_instance_quota_prefix(self) -> torch.Tensor:
        self.sync_placement_to_cpu()
        return self._logical_instance_quota_prefix

    @property
    def rank_quota_prefix(self) -> torch.Tensor:
        return self._rank_quota_prefix

    def get_quota_tensor(self, layer_id: int) -> torch.Tensor:
        self.sync_placement_to_cpu(layer_id)
        return self._logical_instance_quota[layer_id]

    def get_quota_prefix_tensor(self, layer_id: int) -> torch.Tensor:
        self.sync_placement_to_cpu(layer_id)
        return self._logical_instance_quota_prefix[layer_id]

    def get_rank_quota_prefix_tensor(self, layer_id: int) -> torch.Tensor:
        return self._rank_quota_prefix[layer_id]

    def destroy(self):
        assert self.explicitly_destroy

        if self.runtime is not None:
            self.runtime.destroy()
        self.runtime = None

    def allocate_microbatch_slot(self, real_layer_id: int) -> int:
        """Allocate the next virtual layer ID for this real layer.

        Maps ``(real_layer_id, mb_slot)`` → ``virtual_layer_id`` using:
            ``virtual = real_layer_id * max_microbatches + (counter % max_microbatches)``

        The counter wraps modulo ``max_microbatches``, so once a micro-batch's
        backward completes and frees a slot, that slot can be reused by a later
        micro-batch.  The caller must ensure ``max_microbatches`` is at least as
        large as the peak number of in-flight micro-batches per layer.

        For DDP-only (``max_microbatches == 1``), the virtual ID equals the real
        layer ID and the counter overhead is a single integer increment.
        """
        assert real_layer_id < self.real_num_alloc_layers
        mb_slot = self._mb_counters[real_layer_id] % self.max_microbatches
        self._mb_counters[real_layer_id] += 1
        return real_layer_id * self.max_microbatches + mb_slot

    def _real_layer_id(self, virtual_layer_id: int) -> int:
        """Map a virtual layer ID back to the real layer ID."""
        return virtual_layer_id // self.max_microbatches

    def construct_local_master_ptr_pool(
        self,
        layer_id: int,
        fc1_weights: List[torch.Tensor],
        fc2_weights: List[torch.Tensor],
        fc1_grads: Optional[List[torch.Tensor]] = None,
        fc2_grads: Optional[List[torch.Tensor]] = None,
    ):
        assert layer_id < self.real_num_alloc_layers
        assert len(fc1_weights) == self.num_local_master_experts
        assert len(fc2_weights) == self.num_local_master_experts

        def check_tensors_dtype(tensors: List[torch.Tensor], dtype: torch.dtype):
            for t in tensors:
                assert (
                    t.dtype == dtype
                ), f"Expected weight/grad dtype {dtype}, got {t.dtype}"

        check_tensors_dtype(fc1_weights, torch.bfloat16)
        check_tensors_dtype(fc2_weights, torch.bfloat16)

        def _to_dataptr_tensor(tensors: List[torch.Tensor], device) -> torch.Tensor:
            return torch.tensor(
                [t.data_ptr() for t in tensors], dtype=torch.int64, device=device
            ).contiguous()

        gpu_ptr_device = torch.device("cuda", self.device)
        self.local_master_fc1_weight_pool[layer_id] = _to_dataptr_tensor(
            fc1_weights, device="cpu"
        )
        self.local_master_fc2_weight_pool[layer_id] = _to_dataptr_tensor(
            fc2_weights, device="cpu"
        )
        if self.use_gpu_task_build:
            self.local_master_fc1_weight_pool_gpu[layer_id] = _to_dataptr_tensor(
                fc1_weights, device=gpu_ptr_device
            )
            self.local_master_fc2_weight_pool_gpu[layer_id] = _to_dataptr_tensor(
                fc2_weights, device=gpu_ptr_device
            )

        if self.is_train:
            assert (
                fc1_grads is not None and fc2_grads is not None
            ), "Grad tensors required in training mode"
            assert len(fc1_grads) == self.num_local_master_experts
            assert len(fc2_grads) == self.num_local_master_experts
            check_tensors_dtype(fc1_grads, torch.float32)
            check_tensors_dtype(fc2_grads, torch.float32)
            self.local_master_fc1_grad_pool[layer_id] = _to_dataptr_tensor(
                fc1_grads, device="cpu"
            )
            self.local_master_fc2_grad_pool[layer_id] = _to_dataptr_tensor(
                fc2_grads, device="cpu"
            )
            if self.use_gpu_task_build:
                self.local_master_fc1_grad_pool_gpu[layer_id] = _to_dataptr_tensor(
                    fc1_grads, device=gpu_ptr_device
                )
                self.local_master_fc2_grad_pool_gpu[layer_id] = _to_dataptr_tensor(
                    fc2_grads, device=gpu_ptr_device
                )

    def grad_reduce(
        self,
        layer_id: int,
        mode: str = "low_sm",
        previous_event: Optional[EventHandle] = None,
        async_finish: bool = False,
    ):
        """Aggregate replica gradients to masters.

        Args:
            layer_id: Virtual layer ID (encodes both real layer and micro-batch
                slot).  Used for placement map lookup in C++.  Master pointer
                pools are looked up by the real layer ID derived from this.
        """
        assert layer_id < self.num_alloc_layers
        real_lid = self._real_layer_id(layer_id)
        assert (
            self.local_master_fc1_grad_pool[real_lid] is not None
            and self.local_master_fc2_grad_pool[real_lid] is not None
        )
        if self.use_gpu_task_build:
            assert (
                self.local_master_fc1_grad_pool_gpu[real_lid] is not None
                and self.local_master_fc2_grad_pool_gpu[real_lid] is not None
            )
        fc1_grad_ptr_pool = (
            self.local_master_fc1_grad_pool_gpu[real_lid]
            if self.use_gpu_task_build
            else self.local_master_fc1_grad_pool[real_lid]
        )
        fc2_grad_ptr_pool = (
            self.local_master_fc2_grad_pool_gpu[real_lid]
            if self.use_gpu_task_build
            else self.local_master_fc2_grad_pool[real_lid]
        )
        event = self.runtime.grad_reduce(
            layer_id,
            fc1_grad_ptr_pool,
            fc2_grad_ptr_pool,
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
            layer_id: Virtual layer ID.  Used for placement map lookup in C++.
                Master pointer pools are looked up by the derived real layer ID.
            previous_event: Optional event to wait for before starting.
            async_finish: If True, return immediately with an event handle.

        Returns:
            EventHandle if async_finish=True, else None.
        """
        assert layer_id < self.num_alloc_layers
        real_lid = self._real_layer_id(layer_id)
        assert (
            self.local_master_fc1_weight_pool[real_lid] is not None
            and self.local_master_fc2_weight_pool[real_lid] is not None
        )
        if self.use_gpu_task_build:
            assert (
                self.local_master_fc1_weight_pool_gpu[real_lid] is not None
                and self.local_master_fc2_weight_pool_gpu[real_lid] is not None
            )
        fc1_weight_ptr_pool = (
            self.local_master_fc1_weight_pool_gpu[real_lid]
            if self.use_gpu_task_build
            else self.local_master_fc1_weight_pool[real_lid]
        )
        fc2_weight_ptr_pool = (
            self.local_master_fc2_weight_pool_gpu[real_lid]
            if self.use_gpu_task_build
            else self.local_master_fc2_weight_pool[real_lid]
        )
        with torch.cuda.nvtx.range(f"Launch weight_sync (layer {layer_id})"):
            event = self.runtime.weight_sync(
                layer_id,
                fc1_weight_ptr_pool,
                fc2_weight_ptr_pool,
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

        Runs the EPLB-style placement algorithm on CPU or GPU depending on
        ``use_gpu_solver``:
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

    def update_placement_sparse(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
    ):
        assert layer_id < self.num_alloc_layers
        self.runtime.update_placement_sparse(layer_id, topk_ids)

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
            self.sync_placement_to_cpu(layer_id)
            max_replica_cnt = self._logical_replica_counts[layer_id].cpu().max().item()

        if backend == "cuda":
            expanded_probs, expanded_routing_map = self._reroute_cuda(
                layer_id, probs, routing_map
            )
        elif backend == "cpu":
            if self.use_quota_solver:
                raise ValueError("CPU reroute is not supported when use_quota_solver=True")
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
                orig_log_expert_loads_by_rank = orig_log_expert_loads.view(
                    self.num_ranks, self.num_local_master_experts
                ).sum(dim=1)
                balanced_phys_expert_loads_by_rank = balanced_phys_expert_loads.view(
                    self.num_ranks, self.num_local_physical_experts
                ).sum(dim=1)
                orig_imbalance = get_max_by_mean(orig_log_expert_loads_by_rank.cpu())
                balanced_imbalance = get_max_by_mean(
                    balanced_phys_expert_loads_by_rank.cpu()
                )
                with open(self.ep_log_path, "a") as f:
                    f.write(
                        f"Layer {layer_id}: Imbalance (max/mean load per rank): {orig_imbalance:.2f} -> "
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

    def reroute_sparse(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
    ):
        assert layer_id < self.num_alloc_layers
        if self.use_quota_solver:
            raise ValueError("Sparse reroute is not supported when use_quota_solver=True")

        if self.log_expert_loads:
            # Compute original logical expert loads before rerouting.
            # global_logical_expert_loads_gpu is already populated by the preceding
            # update_placement_sparse call via nvshmem allreduce, so we reuse it
            # directly instead of recomputing from topk_ids.
            orig_log_expert_loads = (
                self.runtime.get_global_logical_expert_loads_tensor().cpu().clone()
            )
            self.sync_placement_to_cpu(layer_id)
            max_replica_cnt = self._logical_replica_counts[layer_id].cpu().max().item()

        self.runtime.reroute_sparse(layer_id, topk_ids)

        if self.log_expert_loads:
            # After in-place reroute, topk_ids now holds physical expert IDs.
            # Compute per-physical-expert token counts via bincount + all_reduce.
            flat_phys_ids = topk_ids.flatten().to(torch.int64)
            balanced_phys_expert_loads = torch.bincount(
                flat_phys_ids, minlength=self.num_global_physical_experts
            ).to(torch.int32)
            dist.all_reduce(balanced_phys_expert_loads, group=self.group)

            if self.rank == 0:
                orig_log_expert_loads_by_rank = orig_log_expert_loads.view(
                    self.num_ranks, self.num_local_master_experts
                ).sum(dim=1)
                balanced_phys_expert_loads_by_rank = balanced_phys_expert_loads.view(
                    self.num_ranks, self.num_local_physical_experts
                ).sum(dim=1)
                orig_imbalance = get_max_by_mean(orig_log_expert_loads_by_rank.float())
                balanced_imbalance = get_max_by_mean(
                    balanced_phys_expert_loads_by_rank.float()
                )
                with open(self.ep_log_path, "a") as f:
                    f.write(
                        f"Layer {layer_id}: Imbalance (max/mean load per rank): {orig_imbalance:.2f} -> "
                        f"{balanced_imbalance:.2f} ({orig_imbalance / balanced_imbalance:.2f}x) | "
                        f"max #replicas: {max_replica_cnt}\n"
                    )

    def check_tensors_blob_from_cpp(self):
        assert (
            self._physical_to_logical_map.device == torch.device("cpu")
            and self._logical_to_physical_map.device == torch.device("cpu")
            and self._logical_replica_counts.device == torch.device("cpu")
            and self._logical_instance_quota.device == torch.device("cpu")
            and self._logical_instance_quota_prefix.device == torch.device("cpu")
        )
        assert (
            self._physical_to_logical_map.dtype == torch.int32
            and self._logical_to_physical_map.dtype == torch.int32
            and self._logical_replica_counts.dtype == torch.int32
            and self._logical_instance_quota.dtype == torch.int32
            and self._logical_instance_quota_prefix.dtype == torch.int32
        )
        assert (
            self._physical_to_logical_map.shape
            == (
                self.num_alloc_layers,
                self.num_global_physical_experts,
            )
            and self._logical_to_physical_map.shape
            == (self.num_alloc_layers, self.num_global_logical_experts, self.num_ranks)
            and self._logical_replica_counts.shape
            == (
                self.num_alloc_layers,
                self.num_global_logical_experts,
            )
            and self._logical_instance_quota.shape
            == (self.num_alloc_layers, self.num_global_logical_experts, self.num_ranks)
            and self._logical_instance_quota_prefix.shape
            == (self.num_alloc_layers, self.num_global_logical_experts, self.num_ranks)
        )
        assert self._rank_quota_prefix.device == torch.device("cuda", self.device)
        assert self._rank_quota_prefix.dtype == torch.int32
        assert self._rank_quota_prefix.shape == (
            self.num_alloc_layers,
            self.num_global_logical_experts,
            self.num_ranks,
        )
        assert self.local_replica_weight_buffer.device == torch.device(
            "cuda", self.device
        )
        assert self.local_replica_weight_buffer.dtype == torch.bfloat16
        assert self.local_replica_weight_buffer.shape == (
            self.num_local_redundant_experts,
            self.expert_total_numel,
        )
        if self.is_train:
            assert self.local_replica_grad_buffer.device == torch.device(
                "cuda", self.device
            )
            assert self.local_replica_grad_buffer.dtype == torch.float32
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
