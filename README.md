# UltraEP

UltraEP is an online expert load balancing runtime for large-scale MoE training and inference. It keeps expert placement, token reroute, weight synchronization, and gradient reduction on device so placement updates can run once per layer and per microbatch without pulling the hot path back to the CPU.

## What It Does

UltraEP manages four pieces of the MoE runtime:

- `update_placement`: updates expert placement from live router load statistics.
- `reroute`: expands logical routing to physical routing for dense `[T, L]` routing maps.
- `reroute_sparse`: rewrites sparse `topk_ids` in place from logical IDs to physical IDs.
- `weight_sync` / `grad_reduce`: synchronize replica weights and gradients inside an NVLink domain.

The current codebase has two placement modes:

- Default placement: quota-aware on-device placement.
- Legacy placement: CPU placement behind `Manager(..., legacy_placement=True)`.

Everything outside the legacy placement implementation stays on device. Placement
maps exposed by the Python manager are CUDA tensors; there is no host mirror or
implicit D2H/H2D synchronization in the quota path.

## Current Runtime Model

### Placement

By default, `update_placement()` uses the quota-aware device placement kernel. It consumes live expert loads, computes replica placement, and materializes:

- `physical_to_logical_map`
- `logical_to_physical_map`
- `logical_replica_counts`
- `logical_instance_quota`
- `logical_instance_quota_prefix`
- `rank_quota_prefix`

If you need the old CPU placement path for compatibility, create the manager with:

```python
manager = ultra_ep.Manager(
    group=dist.group.WORLD,
    num_layers=48,
    num_local_master_experts=4,
    num_local_redundant_experts=2,
    expert_fc1_numel=3072 * 4096,
    expert_fc2_numel=1536 * 4096,
    legacy_placement=True,
)
```

That legacy path keeps the public interface identical to the default path: inputs and outputs are still device tensors, and the extra D2H/H2D traffic is contained inside the legacy placement implementation.

### Dense Reroute

Dense reroute always runs on device:

- default placement uses quota-aware dense reroute
- legacy placement uses round-robin dense reroute

Autograd is handled by a thin Python wrapper over the C++ runtime:

- forward builds `expanded_probs` and `expanded_routing_map`
- backward gathers gradients from physical space back to logical space

### Sparse Reroute

Sparse reroute rewrites `topk_ids` in place:

- default placement uses quota-aware sparse reroute
- legacy placement uses round-robin sparse reroute

### Weight Sync And Grad Reduce

Both kernels always use the device task-build path. The task planner reads the current placement directly from device memory and launches the persistent kernels without any CPU-side task construction.

## Build

### Requirements

- SM90 or SM100 GPUs
- NVSHMEM
- CUDA 12.x or 13.x

Install NVSHMEM:

```bash
# CUDA 12.x
pip install nvidia-nvshmem-cu12

# CUDA 13.x
pip install nvidia-nvshmem-cu13
```

Build the extension:

```bash
# Editable / local build
python setup.py build_ext --inplace

# Or build a wheel
python setup.py bdist_wheel
```

If you built NVSHMEM from source, set `NVSHMEM_DIR=/path/to/nvshmem/install`.

## Python API

```python
import torch
import torch.distributed as dist
import ultra_ep

manager = ultra_ep.Manager(
    group=dist.group.WORLD,
    num_layers=48,
    num_local_master_experts=4,
    num_local_redundant_experts=2,
    expert_fc1_numel=3072 * 4096,
    expert_fc2_numel=1536 * 4096,
    is_train=True,
    max_microbatches=1,
)

manager.update_placement(layer_id, routing_map)
expanded_probs, expanded_routing_map = manager.reroute(layer_id, probs, routing_map)
manager.reroute_sparse(layer_id, topk_ids)
manager.weight_sync(layer_id, async_finish=False)
manager.grad_reduce(layer_id, async_finish=False)
```

Notes:

- `Manager.reroute(..., backend="cuda")` is kept for compatibility, but only `"cuda"` is supported.
- `physical_to_logical_map`, `logical_to_physical_map`, `logical_replica_counts`,
  `logical_instance_quota`, and `logical_instance_quota_prefix` are device tensors.
  Reduce diagnostics on GPU and only materialize final scalar summaries for logging.

## Tests

There are two test entrypoints:

```bash
# Single-GPU solver and metric test. Defaults: 64 simulated ranks, 128 experts,
# NVL domain size 64, zipf imbalance ratios 0/1.5/2/2.5/3.
python tests/test_solving.py

# Distributed end-to-end test. Defaults to the launched world size; on the dev
# node use 4 ranks.
torchrun --standalone --nproc_per_node=4 tests/test_e2e.py

# Include HybridEP token dispatch/combine in the e2e report.
torchrun --standalone --nproc_per_node=4 tests/test_e2e.py --include-token-a2a
```

## Configuration

Algorithm and kernel tuning now live entirely on the Python side and are read from `ULTRA_EP_*` environment variables when the manager is created.

### Placement And Reroute

- `ULTRA_EP_BALANCE_THRESHOLD`  
  Default: `1.0`  
  Early-stop threshold for placement replication.

- `ULTRA_EP_QUOTA_LOCALITY_AWARE`  
  Default: `1`  
  Enables locality-aware per-rank quota decomposition.

- `ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA`  
  Default: `1024`  
  Lower bound used by the quota placement kernel when allocating replicas.

- `ULTRA_EP_QUOTA_ALLOW_ZERO_MASTER_QUOTA`  
  Default: `0`  
  Allows the master instance of a logical expert to receive zero quota.

- `ULTRA_EP_QUOTA_ORACLE_EPS`  
  Default: `0.01`  
  Numerical tolerance for the quota placement oracle.

- `ULTRA_EP_QUOTA_KERNEL_STAGE`  
  Default: `1`  
  Selects the supported quota placement kernel stage.

- `ULTRA_EP_QUOTA_REROUTE_INTERLEAVE`  
  Default: `1`  
  Enables deterministic quota interleaving inside quota-aware dense reroute.

### Communication Planning

- `ULTRA_EP_GRAD_REDUCE_NUM_SMS`  
  Default: `24`  
  Number of SMs reserved for the persistent grad-reduce kernel. Must be positive and even.

- `ULTRA_EP_GRAD_REDUCE_DETERMINISTIC`
  Default: `0`
  Enables the deterministic non-atomic grad-reduce path. Use more SMs to alleviate performance drop.

- `ULTRA_EP_WEIGHT_SYNC_PLAN_MODE`  
  Default: `adaptive`  
  Supported values: `direct`, `adaptive`, `force_relay`.

- `ULTRA_EP_WEIGHT_SYNC_RELAY_MIN_REPLICAS`  
  Default: `6`  
  Minimum replica count before adaptive relay becomes eligible.

- `ULTRA_EP_WEIGHT_SYNC_RELAY_MAX_RELAYS`  
  Default: `8`  
  Maximum number of relays used by staged weight sync.

- `ULTRA_EP_WEIGHT_SYNC_RELAY_MIN_FANOUT_GAIN`  
  Default: `2`  
  Minimum expected fanout improvement required before adaptive relay is used.

## Tests

The remaining tests focus on the current runtime surface instead of the removed standalone solver classes:

```bash
# Reroute
torchrun --nproc_per_node=4 tests/test_reroute.py

# Weight sync
torchrun --nproc_per_node=4 tests/test_weight_sync.py

# Grad reduce
torchrun --nproc_per_node=4 tests/test_grad_reduce.py
```

## Notes

- Dense and sparse reroute are explicitly separated.
- Device placement is the default behavior; only CPU code carries an explicit legacy marker.
- The `.cu` translation units avoid including Torch headers directly to keep compile time under control.
