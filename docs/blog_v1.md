---
layout: default
title: "UltraEP: Near-Optimal Load Balancing for Large-Scale MoE Training and Inference"
description: "UltraEP blog: near-optimal load balancing for large-scale MoE training and inference."
lang: en
permalink: /
---

# UltraEP: Near-Optimal Load Balancing for Large-Scale MoE Training and Inference

<section class="blog-abstract" aria-label="Abstract">
  <p>When developing MoE training and inference frameworks, we often assume an <strong>idealized scenario</strong>: post-routing expert load has been force-balanced, so every device receives a near-identical number of tokens and the downstream communication and computation naturally run at full utilization.</p>
  <p>Real training and inference, however, tell a different story. Even when the algorithm side introduces an auxiliary loss or routing bias to preserve model quality, load balance holds mostly in a statistical sense; at the granularity of individual microbatches, hot experts still shift frequently, producing a gap of <strong>up to 2×</strong> between the idealized result and the achieved training/inference throughput.</p>
  <p>As the first system to achieve <strong>exact-load, real-time</strong> balancing, UltraEP turns this idealized assumption into a deployable system capability: within every layer of every microbatch, it replicates hot experts and reroutes tokens on the fly according to the exact load. Building on scale-up connectivity and deep optimization of both the control plane and the data plane, UltraEP keeps its planning and communication overhead on the critical path within <strong>300 µs</strong>. UltraEP thus eliminates load imbalance at the system level.</p>
  <p>On mainstream MoE models from 106B to 671B parameters, UltraEP reaches <strong>94.3%</strong> of the force-balanced ideal performance on average, a <strong>1.49×</strong> improvement over state-of-the-art training and inference frameworks, and reduces inter-rank load imbalance (max : mean) from 1.30–4.01 to <strong>1.01–1.04</strong>. UltraEP has also been deployed in production-scale training.</p>

<figure class="blog-figure blog-figure-full">
  <img src="assets/images/blog-eval_highlights.png" alt="EP overview" width="2737" height="1093" loading="eager" decoding="async" fetchpriority="high">
  <figcaption>UltraEP substantially improves the training and inference throughput of mainstream MoE models, approaching near-ideal performance.</figcaption>
</figure>
</section>

## The Ideal–Reality Gap in Large-Scale MoE Training and Inference

As LLM parameter counts push toward the trillion scale, Mixture-of-Experts (MoE) models have become the dominant architecture: their sparse activation preserves model quality while substantially reducing training and inference cost.

To exploit this architecture, **expert parallelism** (EP) is widely adopted for MoE deployment: experts are distributed across devices, and tokens are exchanged among experts via all-to-all communication. As MoE parameter counts keep growing and inter-GPU bandwidth improves, large-scale expert parallelism (e.g., 64-way or higher) is increasingly common in production.

**Expert load imbalance** is the key factor limiting the real-world performance of expert parallelism. Because of routing dynamics, the instantaneous load on different experts and devices—the total number of tokens received—is inherently uneven. This causes expert computation stragglers, token all-to-all bottlenecks, and activation memory spikes. For a fixed total expert count, the number of experts placed on each rank shrinks as EP grows, making imbalance harder to smooth out; large-scale expert parallelism therefore amplifies the problem further.

<figure class="blog-figure blog-figure-full">
  <img src="assets/images/blog-bg-ep.png" alt="EP overview" width="3134" height="1022" loading="lazy" decoding="async">
  <figcaption>MoE expert parallelism (EP) and load imbalance: 4 experts, EP2, top-k = 2.</figcaption>
</figure>

Among existing solutions, the most representative is [EPLB (DeepSeek, 2025)][EPLB]: it periodically adjusts the placement of all experts across EP ranks based on routing history from a previous time window. The effectiveness of such predictive methods therefore hinges on the stationarity of expert load.

However, we observe that for today's mainstream fine-grained MoE models (with hundreds of "small" experts), expert load is often **highly dynamic** in real training and inference workloads. This makes hot/cold-expert predictions inaccurate, which sharply degrades the effectiveness of balancing operations.

## Core Idea: Real-Time Expert Balancing on Exact Load

UltraEP takes a seemingly aggressive but most direct approach: based on the **exact** post-gating load, it rebalances experts in **real time** within every layer of every microbatch.

This approach obviously eliminates prediction-induced error, but its challenge is equally obvious: predictive methods can hide or amortize the overhead of balancing operations ahead of time, whereas UltraEP exposes this overhead on **critical path** and rebalances at the finest **microbatch** granularity.

<figure class="blog-figure blog-figure-full">
  <img src="assets/images/blog-overview.png" alt="UltraEP overview" width="1690" height="924" loading="lazy" decoding="async">
  <figcaption>UltraEP versus predictive balancing schemes such as EPLB, along three axes: load fidelity, decision timing, and balancing frequency.</figcaption>
</figure>

Building on one **communication premise** and a series of **control-plane and data-plane optimizations**, UltraEP achieves **near-optimal** balancing quality and end-to-end performance with minimal hot-path overhead.

First, UltraEP confines expert-replication traffic to the high-bandwidth scale-up domain, avoiding cross-node expert movement. This is the communication premise for achieving expert replication at the hundred-microsecond scale. Moreover, emerging **rack-scale nodes (RSNs)** significantly expand the scale-up domain, so an entire EP group can fit within a high-bandwidth interconnect.

Given intra-node communication, UltraEP designs an efficient online solver for the balancing plan, together with a set of highly optimized expert weight/gradient communication operators, to minimize the cost of the balancing operations themselves.

## Key Design: Bringing Optimal Load Balancing into Production

UltraEP is positioned as a **production-grade** expert-balancing library, following these design principles:

- **Standalone Python/CUDA runtime**: UltraEP handles redundant-expert state management, balancing-plan solving, and the associated expert transfers, fully decoupled from token all-to-all communication libraries such as DeepEP. Integrating it into mainstream training/inference frameworks requires only a few hundred lines of code changes.
- **GPU-native computation/communication**: all solving and communication operations in UltraEP are on-device, avoiding host-side data synchronization and preserving compatibility with CUDA graphs.
- **Equivalence and generality**: UltraEP does not alter the mathematical equivalence of MoE computation, and includes a series of compatibility adaptations for mainstream parallelism strategies (DP/PP/VPP) and mechanisms such as activation checkpointing and FP8 quantization.
- **Efficient memory management**: through fine-grained management of expert state, UltraEP keeps its extra memory overhead to a minimum, with no dynamic memory allocation at runtime.

On top of these principles, we summarize UltraEP's key designs along four dimensions.

### 1. Expert State Layout: Memory-Friendly Cross-Layer Buffer Reuse

On top of the framework's original expert layout, UltraEP reserves a fixed number of slots on each EP rank for placing replicated **redundant experts**. Unlike the model's inherent **main experts**, redundant experts need not maintain optimizer state, because their gradients are reduced back to the main experts in real time during backpropagation, and optimizer updates are applied only to main experts.

For redundant-expert weight and gradient buffers, UltraEP adopts **cross-layer reuse**, which saves memory dramatically. On Qwen3-235B, for example, the extra memory introduced by a single redundant-expert slot on each EP rank drops from 9.9 GB to 108 MB. This cross-layer reuse is also consistent with UltraEP's per-layer real-time operation.

<figure class="blog-figure blog-figure-full">
  <img src="assets/images/blog-expert-layout.png" alt="Expert layout" width="1382" height="628" loading="lazy" decoding="async">
  <figcaption>UltraEP expert memory management: 8 main experts, EP4, one redundant-expert slot per rank.</figcaption>
</figure>

### 2. End-to-End Integration: Efficient Orchestration of Compute and Communication

Since UltraEP is a real-time balancer, we need to clarify which of the extra computation and communication operations it introduces are exposed on the critical path and which can be hidden. We show forward/backward pass diagrams alongside an actual profiling slice from Qwen3-235B EP64 training, focusing on the deployed effect.

In the forward pass, due to data dependency, UltraEP can only perform **replication planning** and **weight distribution** after gating finishes and the global load information is available. **Reroute** redirects tokens across the multiple replicas of a given main expert. Compared with the first two operations, reroute is lighter and can largely be hidden behind weight distribution.

Optimizations for replication planning (control plane) and weight distribution (data plane) are covered in the next section; these two hot-path operations saturate GPU SM resources to maximize hardware utilization. Overall, based on the EP64 results, UltraEP keeps its extra critical-path overhead within 300 µs, only 1%–2% of the end-to-end time.

<figure class="blog-figure blog-figure-full">
  <img src="assets/images/blog-timeline-fwd.png" alt="FWD timeline" width="1232" height="392" loading="lazy" decoding="async">
  <figcaption>In the forward pass, replication planning and the actual weight transfer fall on the critical path.</figcaption>
</figure>

<figure class="blog-figure blog-figure-full">
  <img src="assets/images/blog-nsys-fwd.png" alt="FWD nsys timeline" width="2761" height="434" loading="lazy" decoding="async">
  <figcaption>On Qwen3-235B, the total critical-path overhead stays largely within 300 µs.</figcaption>
</figure>

In the backward pass, dual to the forward, UltraEP must restore the redundant-expert placement to the state of the corresponding forward microbatch before computing the MoE Dgrad (data gradient), and, before the next MoE Wgrad (weight gradient) computation begins, reduce the Wgrad of all expert replicas in the current layer back into the gradient buffers of their respective main experts—because the redundant-expert gradient buffers are also reused across layers.

Users can flexibly control the number of SMs these communication operations occupy via environment variables, ensuring they are fully hidden behind computation. The next section explains how we prevent these communications from slowing down the compute-intensive backward pass, and how we guarantee the determinism of gradient reduction.

<figure class="blog-figure blog-figure-full">
  <img src="assets/images/blog-timeline-bwd.png" alt="BWD timeline" width="1368" height="392" loading="lazy" decoding="async">
  <figcaption>In the backward pass, expert-weight redistribution<sup id="backward-overlap-ref"><a href="{{ page.url | relative_url }}#backward-overlap-note">1</a></sup> and expert-replica gradient reduction can overlap with other backward computation.</figcaption>
</figure>

<figure class="blog-figure blog-figure-full">
  <img src="assets/images/blog-nsys-bwd.png" alt="BWD nsys timeline" width="2359" height="456" loading="lazy" decoding="async">
  <figcaption>On Qwen3-235B with EP64, the gradient-reduction communication is fully hidden behind the backward computation of the router and attention, without slowing down that computation.</figcaption>
</figure>

### 3. Core Operators: Extreme Control-Plane and Data-Plane Optimization

**3.1 Control plane.** The challenge is how to quickly solve a high-quality expert-balancing plan from the real-time load input. Among existing methods, EPLB uses LPT greedy to solve expert placement but cannot account for real-time token reroute. [LPLB (DeepSeek, 2025)][LPLB] targets the gap between EPLB's periodic expert re-placement and the real-time load by adjusting reroute online via linear programming, but its effectiveness is severely weakened by solvability constraints and EPLB's placement bias. Moreover, in the large-scale expert parallelism we target, the solution space for expert replication is exponentially enlarged.

To this end, we propose an efficient **quota-driven balancing algorithm**. Rather than solving expert placement and token reroute separately, UltraEP directly solves the post-reroute load received by each expert instance, i.e., the quota, thereby coupling the two stages: each exploration step can both instantiate a new expert replica and update the final load distribution.

The overall algorithm framework performs an efficient binary search over the optimal global load-balancing threshold. The quota design fully exploits redundant-expert capacity, achieving the best balancing quality with as few expert replicas as possible. Leveraging warp-level parallelism and reduction, we implement an efficient quota-solving kernel on the GPU that still keeps solving time within 100 µs at EP64.

**3.2 Data plane.** The two communication patterns—expert weight distribution and gradient reduction—are both highly **dynamic and sparse**. Shifts in the hot/cold expert distribution cause the communication plan to change with expert placement at every layer and every microbatch. Classic communication libraries built for regular collectives lack optimizations for such irregular patterns, and in-switch compute-offload schemes such as NVLink SHARP cannot support runtime-dynamic, partial communication groups.

In addition, since a few hot experts may have many replicas while most cold experts need not be replicated at all, the high-volume outbound multicast from the ranks hosting hot experts becomes a new communication bottleneck. Under highly imbalanced load, this severely degrades weight-distribution performance and can even cancel out the gains from load balancing itself.

To fully utilize the physical scale-up bandwidth, UltraEP first builds on **persistent kernels** and shared-memory double-buffering: it splits expert weights or gradients into tiles and uses memory semantics and TMA for asynchronous inter-device data movement.

To eliminate communication hotspots, UltraEP designs a **chunk streaming relay** communication strategy that builds a two-stage relay tree in real time based on the traffic distribution, letting low-traffic ranks share and forward the outbound traffic of hot ranks. By streaming chunks (groups of consecutive tiles) forward instead of waiting for an entire expert transfer to complete, this strategy avoids expensive global barriers and communication bubbles.

<figure class="blog-figure blog-figure-narrow">
  <img src="assets/images/blog-tech-relay.png" alt="Relay tech" width="1348" height="1254" loading="lazy" decoding="async">
  <figcaption>Relay scheme for hot-expert multicast: one expert on rank 0 is replicated to ranks 1–9, with ranks 2, 5, and 8 selected as relays. The figure shows the send/receive channel states of the source and relay ranks along the timeline.</figcaption>
</figure>

For UltraEP's backward communication operators, to avoid resource contention with computation, we build on the efficient persistent-kernel implementation and carefully control SM occupancy and shared-memory footprint so that communication and compute thread blocks can be scheduled on the same SM. Furthermore, to guarantee deterministic floating-point gradient reduction, we replace atomic adds with order-preserving accumulation, minimizing the performance overhead of the deterministic implementation.

### 4. Visualizing the Effect: Tracking Per-Microbatch Balancing Gains

In MoE load-balancing optimization, one easily overlooked issue is how to intuitively evaluate the gains from balancing. The performance metrics of existing training/inference frameworks tend to revolve around end-to-end behavior—MFU and TFLOPS in training, TTFT and TPOT in inference—from which users cannot infer the load state of specific EP ranks and experts.

We therefore introduce a lightweight runtime expert-load profiler in UltraEP that collects, in real time, the load of all experts before and after token reroute, per layer, per microbatch, and per EP group. Although it is not CUDA-graph compatible, by fusing metadata-processing kernels with asynchronous D2H transfers, the profiler's runtime overhead is nearly negligible.

Through the HTML visualization toolchain provided by UltraEP, users can perform *hierarchical* analysis of expert load before and after balancing: they can view the global balancing-degree distribution statistics at a glance, and zoom in to see the specific hot/cold ranks and expert loads within each microbatch.

<figure class="blog-figure blog-figure-full">
  <video autoplay loop muted playsinline preload="none" width="1267" height="697" aria-label="Profiler">
    <source src="assets/images/blog-profiler.mp4" type="video/mp4">
  </video>
  <figcaption>UltraEP hierarchical load analysis: overall distribution and microbatch detail.</figcaption>
</figure>

Together, this profiler and visualization toolchain let users comprehensively assess the effectiveness of a balancing algorithm and locate remaining bottlenecks. UltraEP's interface also supports pluggable balancing algorithms, making it easy to evaluate different designs for a given scenario. Going forward, we will add framework-side profiling of the real MoE compute and communication time, to further clarify the end-to-end improvement from load balancing.

## Experimental Results

We evaluate UltraEP on realistic training and inference scenarios with representative MoE models. Training uses EP64, further scaled by DP or PP at the outer level. Inference focuses on prefill, using EP64 or EP40 depending on the model's expert count. For training, we first fully pretrain three models—GLM4.5-106B, Qwen3-235B, and DeepSeek-V3—then load mid-to-late training checkpoints and continue training under different balancing strategies. For inference, we construct requests from datasets such as LongBench, Codeforces, and DAPO-Math-17K. Training and inference are built on Megatron-LM and SGLang, respectively.

<figure class="blog-figure blog-figure-full">
  <img src="assets/images/blog-eval_train_e2e.png" alt="Results training" width="3296" height="1839" loading="lazy" decoding="async">
  <figcaption>Training results, including throughput (TFLOPS/GPU) and overall balancing degree.</figcaption>
</figure>

<figure class="blog-figure blog-figure-full">
  <img src="assets/images/blog-eval_serving_e2e.png" alt="Results serving" width="4139" height="1771" loading="lazy" decoding="async">
  <figcaption>Inference results, including TTFT as a function of requests per second (RPS), and overall balancing degree.</figcaption>
</figure>

In training, UltraEP reaches 94.6% of ideal throughput on average, a 42% improvement over Megatron-LM. In the more dynamic inference prefill, UltraEP still reaches 90%–97% of ideal throughput, a 1.56× improvement over SGLang. Due to stale historical load and limited replication budgets, EPLB and LPLB consistently lag UltraEP in both balancing quality and end-to-end performance.

UltraEP steadily compresses inter-rank imbalance to 1.01–1.04 in both training and inference. Because UltraEP addresses overall inter-rank load imbalance rather than inter-expert imbalance, the remaining gap to the force-balanced performance ceiling comes mainly from the non-uniform load that real routing places on individual experts in MoE communication and computation, plus the extra control overhead of adding redundant experts—not from residual imbalance or critical-path overhead.

<figure class="blog-figure blog-figure-full">
  <img src="assets/images/blog-eval_prod.png" alt="Results production" width="1553" height="478" loading="lazy" decoding="async">
  <figcaption>In a production task, the loss and throughput over the pretraining of a 288B-parameter MoE model.</figcaption>
</figure>

UltraEP has been deployed in production-scale training. In the full pretraining of a 288B-parameter MoE model, UltraEP sustained no less than 92% of ideal performance, substantially improving and stabilizing long-horizon training throughput.

## Conclusion

UltraEP's core thesis is that, as communication bandwidth improves, real-time, exact system-side load balancing will become a fundamental capability of expert parallelism. Algorithm-side balancing is responsible for training stability and expert specialization, while system-side balancing flattens the load skew that has already occurred within each microbatch; the two have different goals but compose naturally.

From a production perspective, UltraEP's significance lies in eliminating the performance gap between MoE framework development and real training/inference. It encapsulates a series of balancing operations inside the runtime and drives the critical-path overhead low enough that the performance of large-scale expert parallelism stays close to ideal even under real, dynamic load.

A natural next step is the post-training scenario. LLM reinforcement learning (RL) typically alternates between training and rollout (akin to batched inference). Because RL training targets domain-specific data (e.g., code, math) and lacks the algorithm-side balancing regulation of pretraining, expert load often exhibits the strong dynamics similar to inference prefill, as observed in [ReLibra (Jin et al., 2026)][ReLibra]. UltraEP therefore has the opportunity to become a unified load-regulation layer in MoE RL infrastructure.

---

<p class="blog-note" id="backward-overlap-note"><small><a href="{{ page.url | relative_url }}#backward-overlap-ref">1.</a> Because overlapping weight distribution with the MoE Wgrad computation would require intrusively inserting synchronization into the fused grouped-GEMM kernel, in our open-source Megatron-LM patch we place this operation before the MoE Wgrad computation, saturating the SMs as in the forward pass to ensure minimal latency.</small></p>

## Paper and Citation

**Paper:** [UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing](https://arxiv.org/abs/2606.04101)

**Authors:** **Xinming Wei**<sup>1\*</sup>, Chao Jin<sup>1</sup>, Tuo Dai<sup>2</sup>, Yinmin Zhong<sup>1</sup>, Shan Yu<sup>3</sup>, Chengxu Yang<sup>4</sup>, Bingyang Wu<sup>1</sup>, Zili Zhang<sup>1</sup>, Jing Mai<sup>1</sup>, Qianchao Zhu<sup>4</sup>, Zhouyang Li<sup>4</sup>, Yuliang Liu<sup>4†</sup>, Guojie Luo<sup>1†</sup>

<sup>1</sup>Peking University &nbsp; <sup>2</sup>Xiaohongshu Inc. &nbsp; <sup>3</sup>Shanghai AI Laboratory &nbsp; <sup>4</sup>Independent Researcher

*\*Work done during an internship at Xiaohongshu Inc. †Corresponding authors.*

If UltraEP helps your research or development, please cite:

```bibtex
@article{wei2026ultraep,
  title={UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing},
  author={Xinming Wei and Chao Jin and Tuo Dai and Yinmin Zhong and Shan Yu and Chengxu Yang and Bingyang Wu and Zili Zhang and Jing Mai and Qianchao Zhu and Zhouyang Li and Yuliang Liu and Guojie Luo},
  journal={arXiv preprint arXiv:2606.04101},
  year={2026},
  doi={10.48550/arXiv.2606.04101}
}
```

[EPLB]: https://github.com/deepseek-ai/EPLB
[LPLB]: https://github.com/deepseek-ai/LPLB
[ReLibra]: https://arxiv.org/abs/2605.08639
