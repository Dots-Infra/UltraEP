# UltraEP: Online Expert Load Balancing for Large-Scale MoE Serving and Training

UltraEP is a high-performance, online expert load balancing library specifically designed for Mixture-of-Experts (MoE) training and inference. It provides efficient synchronization of expert weights and gradients across GPUs, enabling flexible expert placement and redundant expert strategies to mitigate load imbalance. It incurs near-zero latency or memory overhead with dedicated kernels and layer-reused replica weight/gradient buffers.

## 🚀 Roadmap

- [x] Intra-NVLINK domain master-replica synchronization.
- [x] High-performance `weight_sync` and `grad_reduce` kernels.
- [x] Support for SM90 (H100/H800) and SM100 (Blackwell) architectures.
- [x] Advanced online expert placement and token dispatch algorithms.
- [x] Deep integration with mainstream training and inference frameworks (e.g., Megatron-LM, SGLang, vLLM).
- [ ] Support for FP8 training/serving.
- [ ] C2C expert offloading for reduced NVLink traffic contention.
- [ ] CUDA graph integration with pure on-device kernels.
- [ ] Support for cross-RDMA (Inter-node) expert synchronization.

## 💡 Background & Concepts

In MoE models, token distribution across experts can be highly skewed, leading to computational bottlenecks on certain "hot" experts. UltraEP addresses this by allowing experts to be replicated across different GPUs.

### Key Concepts

- **Logical Experts**: The experts as defined in the model architecture (e.g., E0, E1, ..., E7).
- **Physical Experts**: The actual expert instances stored on physical GPUs.
- **Master Expert**: The primary physical instance of a logical expert, responsible for maintaining the authoritative weights and optimizer states.
- **Redundant (Replica) Expert**: Additional physical instances of a logical expert, used to share the computation load. They only store weights and gradients shared by layers without optimizer states.

![EPLB Modeling](images/eplb_modeling.png)

### Data Structures

UltraEP uses layer-wise mappings to delineate the state of expert placement:
- `physical_to_logical_map`: Maps each physical expert on a GPU to its corresponding logical expert ID.
- `logical_to_physical_map`: Maps each logical expert to its master and replica physical locations.
- `logical_replica_counts`: Tracks the total number of physical instances (master + replicas) for each logical expert.

## 🛠️ Setup

### Prerequisites
- Hardware: Only support SM90 and SM100 GPUs.
- Dependencies:
  - `nvshmem`: High-performance communication library for NVIDIA GPUs.
  ```bash
  # For CUDA 12.x
  pip install nvidia-nvshmem-cu12
  # For CUDA 13.x
  pip install nvdia-nvshmem-cu13==3.4.5
  ```

### Build and Install
```bash
# Clone the repository
git clone https://github.com/your-repo/UltraEP.git
cd UltraEP

# Build the project
# CUDA 12.x
./build.sh
# For CUDA 13.x, you need to compile NVSHMEM v3.4.5 from source first,
# then set the installation path
NVSHMEM_DIR=/path/to/nvshmem/install python setup.py bdist_wheel

# Install the generated wheel
pip install dist/*.whl
```

## 📖 Usage

UltraEP currently provides two primary operators for managing master-replica synchronization. Both sync and async modes are supported for flexible overlapping control:

### 1. `weight_sync` (CUDA)
Used during **inference** or the **forward** pass of training. It broadcasts the weights from the master expert to all its redundant experts across the NVLINK domain. This should be finished before the MoE computation starts in each layer.

### 2. `grad_reduce` (CUDA)
Used during the **backward** pass of training. It aggregates (reduces) gradients from all redundant experts back to their respective master experts, then zeros replica gradient buffers. Since the replica grad buffer is cross-layer shared, `grad_reduce` must complete before the next layer starts computing expert gradients.
- High-SM mode (`high_sm`): Optimized for maximum throughput when GPU resources are primarily dedicated to this reduction.
- Low-SM mode (`low_sm`): Recommended when you need to overlap the gradient reduction with other backward computations (e.g., Attention or MLP calculations) to hide communication latency.

### 3. `update_placement` (CPU)
Dynamically adjusts expert placement based on real-time load statistics. It runs an EPLB-style greedy replication and bin-packing algorithm on the CPU. The algorithm is deterministic, ensuring all ranks compute identical placements without additional communication.

### 4. `reroute` (CUDA)
A high-performance CUDA kernel that expands token routing from logical experts to physical experts. It uses deterministic round-robin dispatch to distribute tokens among master and replica instances, effectively balancing the computation load across the NVLINK domain.

### Example Code Snippet

```python
import torch
import ultra_ep

# Initialize Manager
manager = ultra_ep.Manager(
    group=dist.group.WORLD,
    num_layers=48,
    num_local_master_experts=4,
    num_local_redundant_experts=2,
    expert_fc1_numel=3072 * 4096,
    expert_fc2_numel=1536 * 4096,
)

# ... (Register master weight/grad buffers) ...

# --- Token Dispatch & Placement ---
# Update placement based on routing map
manager.update_placement(layer_id=layer_x, routing_map=routing_map)

# Reroute tokens to physical experts (master + replicas)
expanded_probs, expanded_routing_map = manager.reroute(
    layer_id=layer_x, probs=probs, routing_map=routing_map
)

# --- Forward Pass ---
# Sync master weights to replicas before MoE calculation
manager.weight_sync(layer_id=layer_x, async_finish=False)
# Run MoE forward using expanded_probs and expanded_routing_map...

# --- Backward Pass ---
# Run MoE backward to get gradients...
# Reduce replica gradients back to masters
manager.grad_reduce(layer_id=layer_x, mode='low_sm', async_finish=False)
```

## 🔍 Hardware Support & Constraints

- NVLINK Domain: Supports automatic detection of NVLINK domain size.
- Architectures: Optimized for SM90 (max NVL size 8) and SM100 (super nodes like NVL72).
- Current Constraint: Synchronization is currently limited to within a single NVLINK domain. Expert placement must ensure that a logical expert's master and all its replicas reside within the same NVLINK domain.

## 🧪 Testing

You can run the provided tests to verify correctness and benchmark performance:

```bash
# Test Weight Synchronization
torchrun --nproc_per_node=4 ..... tests/test_weight_sync.py

# Test Gradient Reduction
torchrun --nproc_per_node=4 ..... tests/test_grad_reduce.py

# Test Expert Placement
python3 tests/test_placement.py --num-ranks 32 --nvl-domain-size 8 \
    --num-local-master 4 --num-local-redundant 2

# Test Reroute Kernel
torchrun --nproc_per_node=4 ..... tests/test_reroute.py \
    --num-local-master 4 --num-local-redundant 2 --T 8192 --topk 8
```

These tests verify numerical correctness against a golden reference and report end-to-end latency and achieved bandwidth, under either uniform or skewed expert placement (the latter might be more common in practice, with hot/cold experts unevenly distributed).
