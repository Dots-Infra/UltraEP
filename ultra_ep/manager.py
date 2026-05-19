import os
import torch
import torch.distributed as dist
from typing import List, Optional
from datetime import datetime

import ultra_ep._C as _C
from .config import load_tuning_from_env
from .runtime import init_runtime
from .event import EventHandle
from .reroute import _DenseRerouteFunction
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
        legacy_placement: bool = False,
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

        # Master weight/grad pointer pools, indexed by real layer.
        self.local_master_fc1_weight_ptr_pool = [None] * self.real_num_alloc_layers
        self.local_master_fc2_weight_ptr_pool = [None] * self.real_num_alloc_layers
        self.local_master_fc1_grad_ptr_pool = [None] * self.real_num_alloc_layers
        self.local_master_fc2_grad_ptr_pool = [None] * self.real_num_alloc_layers

        # Per-real-layer micro-batch slot counters (wraps modulo max_microbatches)
        self._mb_counters = [0] * self.real_num_alloc_layers

        tuning = load_tuning_from_env()
        self.grad_reduce_num_sms = tuning.grad_reduce_num_sms
        self.grad_reduce_deterministic = tuning.grad_reduce_deterministic

        # Expert loads logging
        self.log_expert_loads = tuning.log_expert_loads
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ep_loads_save_dir = tuning.loads_save_dir
        os.makedirs(self.ep_loads_save_dir, exist_ok=True)
        self.ep_log_path = os.path.join(
            self.ep_loads_save_dir, f"ep_loads_{now_str}_group{self.id}.log"
        )

        # Create cpp handle
        self.explicitly_destroy = explicitly_destroy
        self.legacy_placement = legacy_placement
        self.quota_kernel_stage = tuning.quota_kernel_stage
        self.quota_reroute_interleave = tuning.quota_reroute_interleave
        weight_sync_plan_mode = tuning.weight_sync_plan_mode
        weight_sync_plan_mode_id = tuning.weight_sync_plan_mode_id
        if self.nvl_domain_size <= 8 and weight_sync_plan_mode != "direct":
            if self.rank == 0:
                print(
                    "UltraEP: NVL domain size <= 8; forcing weight_sync "
                    f"plan mode to direct (was {weight_sync_plan_mode}).",
                    flush=True,
                )
            weight_sync_plan_mode = "direct"
            weight_sync_plan_mode_id = 0
        self.weight_sync_plan_mode = weight_sync_plan_mode
        self.weight_sync_relay_min_replicas = tuning.weight_sync_relay_min_replicas
        self.weight_sync_relay_max_relays = tuning.weight_sync_relay_max_relays
        self.weight_sync_relay_min_fanout_gain = (
            tuning.weight_sync_relay_min_fanout_gain
        )
        self.reroute_mode = "round_robin" if legacy_placement else "quota"
        self.runtime = _C.Manager(
            self.num_alloc_layers,
            num_local_master_experts,
            num_local_redundant_experts,
            expert_fc1_numel,
            expert_fc2_numel,
            is_train,
            explicitly_destroy,
            legacy_placement,
            tuning.balance_threshold,
            tuning.quota_locality_aware,
            tuning.quota_min_tokens_per_replica,
            tuning.quota_allow_zero_master_quota,
            tuning.quota_oracle_eps,
            tuning.quota_kernel_stage,
            tuning.quota_reroute_interleave,
            self.grad_reduce_num_sms,
            self.grad_reduce_deterministic,
            weight_sync_plan_mode_id,
            tuning.weight_sync_relay_min_replicas,
            tuning.weight_sync_relay_max_relays,
            tuning.weight_sync_relay_min_fanout_gain,
        )
        assert self.runtime.is_available()

        # Placement maps are device-resident. Tests and diagnostics should reduce
        # metrics on GPU and only materialize small scalar summaries for printing.
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

    @property
    def physical_to_logical_map(self) -> torch.Tensor:
        return self._physical_to_logical_map

    @property
    def logical_to_physical_map(self) -> torch.Tensor:
        return self._logical_to_physical_map

    @property
    def logical_replica_counts(self) -> torch.Tensor:
        return self._logical_replica_counts

    @property
    def logical_instance_quota(self) -> torch.Tensor:
        return self._logical_instance_quota

    @property
    def logical_instance_quota_prefix(self) -> torch.Tensor:
        return self._logical_instance_quota_prefix

    @property
    def rank_quota_prefix(self) -> torch.Tensor:
        return self._rank_quota_prefix

    def get_quota_tensor(self, layer_id: int) -> torch.Tensor:
        return self._logical_instance_quota[layer_id]

    def get_quota_prefix_tensor(self, layer_id: int) -> torch.Tensor:
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

        weight_or_grad_device = torch.device("cuda", self.device)
        self.local_master_fc1_weight_ptr_pool[layer_id] = _to_dataptr_tensor(
            fc1_weights, device=weight_or_grad_device
        )
        self.local_master_fc2_weight_ptr_pool[layer_id] = _to_dataptr_tensor(
            fc2_weights, device=weight_or_grad_device
        )

        if self.is_train:
            assert (
                fc1_grads is not None and fc2_grads is not None
            ), "Grad tensors required in training mode"
            assert len(fc1_grads) == self.num_local_master_experts
            assert len(fc2_grads) == self.num_local_master_experts
            check_tensors_dtype(fc1_grads, torch.float32)
            check_tensors_dtype(fc2_grads, torch.float32)
            self.local_master_fc1_grad_ptr_pool[layer_id] = _to_dataptr_tensor(
                fc1_grads, device=weight_or_grad_device
            )
            self.local_master_fc2_grad_ptr_pool[layer_id] = _to_dataptr_tensor(
                fc2_grads, device=weight_or_grad_device
            )

    def grad_reduce(
        self,
        layer_id: int,
        previous_event: Optional[EventHandle] = None,
        async_finish: bool = False,
    ):
        """Aggregate replica gradients to masters.

        Args:
            layer_id: Virtual layer ID (encodes both real layer and micro-batch
                slot).  Used for placement map lookup in C++.  Master pointer
                pools are looked up by the real layer ID derived from this.

        Notes:
            The grad-reduce SM budget is controlled globally via the
            ``ULTRA_EP_GRAD_REDUCE_NUM_SMS`` environment variable.
            Set ``ULTRA_EP_GRAD_REDUCE_DETERMINISTIC=1`` to use the deterministic
            non-atomic path.
        """
        assert layer_id < self.num_alloc_layers
        real_lid = self._real_layer_id(layer_id)
        assert (
            self.local_master_fc1_grad_ptr_pool[real_lid] is not None
            and self.local_master_fc2_grad_ptr_pool[real_lid] is not None
        )
        event = self.runtime.grad_reduce(
            layer_id,
            self.local_master_fc1_grad_ptr_pool[real_lid],
            self.local_master_fc2_grad_ptr_pool[real_lid],
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

        The runtime derives a deterministic communication plan from the current
        placement. Mild cases stay on the flat direct fan-out path; extreme hot
        masters may use a staged relay plan to reduce source-side bottlenecks.

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
            self.local_master_fc1_weight_ptr_pool[real_lid] is not None
            and self.local_master_fc2_weight_ptr_pool[real_lid] is not None
        )
        with torch.cuda.nvtx.range(f"Launch weight_sync (layer {layer_id})"):
            event = self.runtime.weight_sync(
                layer_id,
                self.local_master_fc1_weight_ptr_pool[real_lid],
                self.local_master_fc2_weight_ptr_pool[real_lid],
                getattr(previous_event, "event", None),
                async_finish,
            )
            return EventHandle(event)

    def set_weight_sync_plan_mode(self, plan_mode: str):
        normalized = plan_mode.lower().replace("_", "")
        mode_ids = {"direct": 0, "adaptive": 1, "adaptiverelay": 1, "forcerelay": 2}
        if normalized not in mode_ids:
            raise ValueError(
                "plan_mode must be one of: direct, adaptive_relay, force_relay"
            )
        self.weight_sync_plan_mode = normalized
        self.runtime.set_weight_sync_plan_mode(mode_ids[normalized])

    def set_grad_reduce_deterministic(self, deterministic: bool):
        self.grad_reduce_deterministic = bool(deterministic)
        self.runtime.set_grad_reduce_deterministic(
            self.grad_reduce_deterministic, self.grad_reduce_num_sms
        )

    def update_placement(
        self,
        layer_id: int,
        routing_map: torch.Tensor,
        verify_reduced_loads: bool = False,
    ):
        """
        Update expert placement for a single layer based on real-time load statistics.

        Runs the default device placement algorithm and only falls back to the
        legacy CPU implementation when ``legacy_placement=True``:
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
            max_replica_cnt = self._logical_replica_counts[layer_id].max().item()

        if backend != "cuda":
            raise ValueError(
                "Only backend='cuda' is supported; CPU reroute has been removed"
            )

        expanded_probs, expanded_routing_map = self._dense_reroute(
            layer_id, probs, routing_map
        )

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
                orig_imbalance = get_max_by_mean(orig_log_expert_loads_by_rank)
                balanced_imbalance = get_max_by_mean(balanced_phys_expert_loads_by_rank)
                with open(self.ep_log_path, "a") as f:
                    f.write(
                        f"Layer {layer_id}: Imbalance (max/mean load per rank): {orig_imbalance:.2f} -> "
                        f"{balanced_imbalance:.2f} ({orig_imbalance / balanced_imbalance:.2f}x) | "
                        f"max #replicas: {max_replica_cnt}\n"
                    )

        return expanded_probs, expanded_routing_map

    def _dense_reroute(
        self,
        layer_id: int,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ):
        """Dense reroute through the fused device kernel."""
        if self.is_train and probs.requires_grad:
            expanded_probs, expanded_routing_map = _DenseRerouteFunction.apply(
                probs,
                routing_map,
                self.runtime,
                layer_id,
            )
        else:
            with torch.cuda.nvtx.range(f"Dense reroute forward (layer {layer_id})"):
                expanded_probs, expanded_routing_map = (
                    self.runtime.dense_reroute_forward(layer_id, probs, routing_map)
                )

        return expanded_probs, expanded_routing_map

    def reroute_sparse(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
    ):
        assert layer_id < self.num_alloc_layers

        if self.log_expert_loads:
            # Recompute from the current sparse routing ids. Decode reroute does
            # not run update_placement_sparse, so the runtime load tensor may be
            # stale from an earlier layer/prefill batch.
            flat_log_ids = topk_ids.flatten()
            valid_log_ids = flat_log_ids[
                (flat_log_ids >= 0) & (flat_log_ids < self.num_global_logical_experts)
            ].to(torch.int64)
            orig_log_expert_loads = torch.bincount(
                valid_log_ids,
                minlength=self.num_global_logical_experts,
            ).to(torch.int32)
            dist.all_reduce(orig_log_expert_loads, group=self.group)
            max_replica_cnt = self._logical_replica_counts[layer_id].max().item()
            l2p = self._logical_to_physical_map[layer_id]
            quota = self._logical_instance_quota[layer_id]
            valid_placement = l2p >= 0
            expected_phys_expert_loads = torch.zeros(
                (self.num_global_physical_experts,),
                dtype=torch.int32,
                device=topk_ids.device,
            )
            expected_phys_expert_loads.scatter_add_(
                0,
                l2p.clamp_min(0).flatten(),
                quota.masked_fill(~valid_placement, 0).flatten(),
            )

        self.runtime.reroute_sparse(layer_id, topk_ids)

        if self.log_expert_loads:
            # After in-place reroute, topk_ids now holds physical expert IDs.
            # Compute per-physical-expert token counts via bincount + all_reduce.
            flat_phys_ids = topk_ids.flatten()
            valid_phys_ids = flat_phys_ids[
                (flat_phys_ids >= 0)
                & (flat_phys_ids < self.num_global_physical_experts)
            ].to(torch.int64)
            balanced_phys_expert_loads = torch.bincount(
                valid_phys_ids, minlength=self.num_global_physical_experts
            ).to(torch.int32)
            dist.all_reduce(balanced_phys_expert_loads, group=self.group)

            if self.rank == 0:
                orig_log_expert_loads_by_rank = orig_log_expert_loads.view(
                    self.num_ranks, self.num_local_master_experts
                ).sum(dim=1)
                balanced_phys_expert_loads_by_rank = balanced_phys_expert_loads.view(
                    self.num_ranks, self.num_local_physical_experts
                ).sum(dim=1)
                expected_phys_expert_loads_by_rank = expected_phys_expert_loads.view(
                    self.num_ranks, self.num_local_physical_experts
                ).sum(dim=1)
                orig_total = int(orig_log_expert_loads.sum().item())
                expected_total = int(expected_phys_expert_loads.sum().item())
                balanced_total = int(balanced_phys_expert_loads.sum().item())
                orig_imbalance = get_max_by_mean(orig_log_expert_loads_by_rank.float())
                expected_imbalance = get_max_by_mean(
                    expected_phys_expert_loads_by_rank.float()
                )
                balanced_imbalance = get_max_by_mean(
                    balanced_phys_expert_loads_by_rank.float()
                )
                with open(self.ep_log_path, "a") as f:
                    f.write(
                        f"Layer {layer_id}: Imbalance (max/mean load per rank): {orig_imbalance:.2f} -> "
                        f"{balanced_imbalance:.2f} ({orig_imbalance / balanced_imbalance:.2f}x) | "
                        f"expected {expected_imbalance:.2f} | "
                        f"tokens {orig_total}/{expected_total}/{balanced_total} | "
                        f"max #replicas: {max_replica_cnt}\n"
                    )

    def check_tensors_blob_from_cpp(self):
        assert (
            self._physical_to_logical_map.device == torch.device("cuda", self.device)
            and self._logical_to_physical_map.device
            == torch.device("cuda", self.device)
            and self._logical_replica_counts.device == torch.device("cuda", self.device)
            and self._logical_instance_quota.device == torch.device("cuda", self.device)
            and self._logical_instance_quota_prefix.device
            == torch.device("cuda", self.device)
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
