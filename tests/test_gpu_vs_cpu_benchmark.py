"""
Benchmark GPU solver vs CPU solver through the public Manager API.

Focuses on true end-to-end EPLB-step latency instead of kernel-only timing:
  1. update_placement() latency via Manager
  2. Full pipeline latency:
       update_placement -> reroute -> weight_sync -> grad_reduce
  3. Scaling trend across different expert configurations

Example:
    torchrun --nproc_per_node=4 tests/test_gpu_vs_cpu_benchmark.py --benchmark all
"""

import argparse
import os
import sys
from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist

try:
    import ultra_ep
except ImportError:
    print("ERROR: Cannot import ultra_ep.", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import bench_stats, generate_routing_map_from_distribution

NUM_LAYERS = 4


def print_rank0(msg: str):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def solver_name(use_gpu_solver: bool) -> str:
    return "GPU" if use_gpu_solver else "CPU"


def validate_distribution(distribution: str):
    if distribution not in ("uniform", "skewed"):
        raise ValueError(
            f"Unsupported distribution: {distribution}. Expected one of: uniform, skewed"
        )


def create_manager(
    group,
    num_layers: int,
    num_local_master: int,
    num_local_redundant: int,
    expert_fc1_numel: int,
    expert_fc2_numel: int,
    use_gpu_solver: bool,
):
    return ultra_ep.Manager(
        group=group,
        num_layers=num_layers,
        num_local_master_experts=num_local_master,
        num_local_redundant_experts=num_local_redundant,
        expert_fc1_numel=expert_fc1_numel,
        expert_fc2_numel=expert_fc2_numel,
        is_train=True,
        explicitly_destroy=True,
        use_gpu_solver=use_gpu_solver,
    )


def validate_topk(num_global_logical_experts: int, topk: int):
    assert topk <= num_global_logical_experts, (
        f"topk={topk} exceeds num_global_logical_experts={num_global_logical_experts}"
    )


def setup_master_ptr_pool(
    manager,
    layer_id: int,
    num_local_master: int,
    expert_fc1_numel: int,
    expert_fc2_numel: int,
    seed: int,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    fc1_weights = [
        torch.randn(expert_fc1_numel, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_local_master)
    ]
    fc2_weights = [
        torch.randn(expert_fc2_numel, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_local_master)
    ]
    fc1_grads = [
        torch.randn(expert_fc1_numel, device="cuda", dtype=torch.float32)
        for _ in range(num_local_master)
    ]
    fc2_grads = [
        torch.randn(expert_fc2_numel, device="cuda", dtype=torch.float32)
        for _ in range(num_local_master)
    ]
    manager.construct_local_master_ptr_pool(
        layer_id, fc1_weights, fc2_weights, fc1_grads, fc2_grads
    )


def format_stats(stats: Dict[str, float]) -> str:
    return (
        f"mean={stats['mean'] * 1e6:8.1f}us  "
        f"p50={stats['p50'] * 1e6:8.1f}us  "
        f"p99={stats['p99'] * 1e6:8.1f}us  "
        f"min={stats['min'] * 1e6:8.1f}us  "
        f"max={stats['max'] * 1e6:8.1f}us"
    )


def speedup(cpu_stats: Dict[str, float], gpu_stats: Dict[str, float]) -> float:
    return cpu_stats["mean"] / gpu_stats["mean"] if gpu_stats["mean"] > 0 else 0.0


def benchmark_update_placement(
    args,
    num_local_master: int,
    num_local_redundant: int,
    distribution: str,
) -> Dict[str, Dict[str, float]]:
    rank = dist.get_rank()
    group = dist.group.WORLD
    results = {}

    for use_gpu_solver in (False, True):
        manager = create_manager(
            group,
            num_layers=NUM_LAYERS,
            num_local_master=num_local_master,
            num_local_redundant=num_local_redundant,
            expert_fc1_numel=args.expert_fc1_numel,
            expert_fc2_numel=args.expert_fc2_numel,
            use_gpu_solver=use_gpu_solver,
        )
        layer_id = 0
        num_global_logical_experts = manager.num_global_logical_experts
        validate_topk(num_global_logical_experts, args.topk)
        validate_distribution(distribution)

        routing_map = generate_routing_map_from_distribution(
            num_tokens=args.num_tokens,
            num_global_logical_experts=num_global_logical_experts,
            topk=args.topk,
            distribution=distribution,
            seed=args.seed + rank,
            num_ranks=dist.get_world_size(),
            num_local_master=num_local_master,
            num_nvl_ranks=manager.nvl_domain_size,
            hot_expert_ratio_per_nvl_domain=args.hot_expert_ratio_per_nvl_domain,
        )

        stats = bench_stats(
            lambda: manager.update_placement(layer_id, routing_map),
            num_warmups=args.warmup_iters,
            num_tests=args.bench_iters,
            use_barrier=True,
        )
        results[solver_name(use_gpu_solver)] = stats
        manager.destroy()
        dist.barrier()

    return results


def benchmark_full_pipeline(
    args,
    num_local_master: int,
    num_local_redundant: int,
    distribution: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    rank = dist.get_rank()
    group = dist.group.WORLD
    results = {}

    for use_gpu_solver in (False, True):
        manager = create_manager(
            group,
            num_layers=NUM_LAYERS,
            num_local_master=num_local_master,
            num_local_redundant=num_local_redundant,
            expert_fc1_numel=args.expert_fc1_numel,
            expert_fc2_numel=args.expert_fc2_numel,
            use_gpu_solver=use_gpu_solver,
        )
        layer_id = 0
        num_global_logical_experts = manager.num_global_logical_experts
        validate_topk(num_global_logical_experts, args.topk)
        validate_distribution(distribution)

        routing_map = generate_routing_map_from_distribution(
            num_tokens=args.num_tokens,
            num_global_logical_experts=num_global_logical_experts,
            topk=args.topk,
            distribution=distribution,
            seed=args.seed + rank,
            num_ranks=dist.get_world_size(),
            num_local_master=num_local_master,
            num_nvl_ranks=manager.nvl_domain_size,
            hot_expert_ratio_per_nvl_domain=args.hot_expert_ratio_per_nvl_domain,
        )
        probs = torch.randn(
            args.num_tokens,
            num_global_logical_experts,
            dtype=torch.float32,
            device="cuda",
        )

        setup_master_ptr_pool(
            manager,
            layer_id=layer_id,
            num_local_master=num_local_master,
            expert_fc1_numel=args.expert_fc1_numel,
            expert_fc2_numel=args.expert_fc2_numel,
            seed=args.seed + rank + (1000 if use_gpu_solver else 0),
        )
        manager.update_placement(layer_id, routing_map)

        replica_weight_buffer = manager.local_replica_weight_buffer
        replica_grad_buffer = manager.local_replica_grad_buffer
        replica_grad_reference = torch.randn_like(replica_grad_buffer)

        stage_stats = {}
        stage_stats["update_placement"] = bench_stats(
            lambda: manager.update_placement(layer_id, routing_map),
            num_warmups=args.warmup_iters,
            num_tests=args.bench_iters,
            use_barrier=True,
        )
        stage_stats["reroute"] = bench_stats(
            lambda: manager.reroute(layer_id, probs, routing_map, backend="cuda"),
            num_warmups=args.warmup_iters,
            num_tests=args.bench_iters,
            use_barrier=True,
        )
        stage_stats["weight_sync"] = bench_stats(
            lambda: manager.weight_sync(layer_id, async_finish=False),
            num_warmups=args.warmup_iters,
            num_tests=args.bench_iters,
            use_barrier=True,
            pre_fn=lambda: replica_weight_buffer.random_(0, 3),
        )
        stage_stats["grad_reduce"] = bench_stats(
            lambda: manager.grad_reduce(
                layer_id, mode=args.grad_reduce_mode, async_finish=False
            ),
            num_warmups=args.warmup_iters,
            num_tests=args.bench_iters,
            use_barrier=True,
            pre_fn=lambda: replica_grad_buffer.copy_(replica_grad_reference),
        )

        def pipeline_pre_fn():
            replica_weight_buffer.random_(0, 3)
            replica_grad_buffer.copy_(replica_grad_reference)

        def pipeline_fn():
            manager.update_placement(layer_id, routing_map)
            manager.reroute(layer_id, probs, routing_map, backend="cuda")
            manager.weight_sync(layer_id, async_finish=False)
            manager.grad_reduce(
                layer_id, mode=args.grad_reduce_mode, async_finish=False
            )

        stage_stats["total"] = bench_stats(
            pipeline_fn,
            num_warmups=args.warmup_iters,
            num_tests=args.bench_iters,
            use_barrier=True,
            pre_fn=pipeline_pre_fn,
        )
        results[solver_name(use_gpu_solver)] = stage_stats
        manager.destroy()
        dist.barrier()

    return results


def print_update_placement_results(
    distribution: str, results: Dict[str, Dict[str, float]]
):
    if dist.get_rank() != 0:
        return
    print_rank0(f"\n[update_placement] distribution={distribution}")
    print_rank0(f"  CPU  {format_stats(results['CPU'])}")
    print_rank0(
        f"  GPU  {format_stats(results['GPU'])}  speedup={speedup(results['CPU'], results['GPU']):.2f}x"
    )


def print_full_pipeline_results(
    distribution: str, results: Dict[str, Dict[str, Dict[str, float]]]
):
    if dist.get_rank() != 0:
        return
    print_rank0(f"\n[full_pipeline] distribution={distribution}")
    for stage in (
        "update_placement",
        "reroute",
        "weight_sync",
        "grad_reduce",
        "total",
    ):
        cpu_stats = results["CPU"][stage]
        gpu_stats = results["GPU"][stage]
        print_rank0(f"  stage={stage}")
        print_rank0(f"    CPU  {format_stats(cpu_stats)}")
        print_rank0(
            f"    GPU  {format_stats(gpu_stats)}  speedup={speedup(cpu_stats, gpu_stats):.2f}x"
        )


def bench_placement_gpu_vs_cpu(args, distributions: Iterable[str]):
    print_rank0("\n" + "=" * 80)
    print_rank0("Benchmark 1.1: update_placement GPU vs CPU")
    print_rank0("=" * 80)
    for distribution in distributions:
        results = benchmark_update_placement(
            args,
            num_local_master=args.num_local_master_experts,
            num_local_redundant=args.num_local_redundant_experts,
            distribution=distribution,
        )
        print_update_placement_results(distribution, results)


def bench_full_pipeline_gpu_vs_cpu(args, distributions: Iterable[str]):
    print_rank0("\n" + "=" * 80)
    print_rank0("Benchmark 1.2: full pipeline GPU vs CPU")
    print_rank0("=" * 80)
    for distribution in distributions:
        results = benchmark_full_pipeline(
            args,
            num_local_master=args.num_local_master_experts,
            num_local_redundant=args.num_local_redundant_experts,
            distribution=distribution,
        )
        print_full_pipeline_results(distribution, results)


def bench_scaling(args):
    print_rank0("\n" + "=" * 80)
    print_rank0("Benchmark 1.3: scaling trend")
    print_rank0("=" * 80)

    masters = parse_int_csv(args.scaling_num_local_master)
    redundants = parse_int_csv(args.scaling_num_local_redundant)
    if len(masters) != len(redundants):
        raise ValueError(
            "scaling_num_local_master and scaling_num_local_redundant must have the same length"
        )

    rows: List[Tuple[int, int, float, float, float]] = []
    for num_local_master, num_local_redundant in zip(masters, redundants):
        results = benchmark_full_pipeline(
            args,
            num_local_master=num_local_master,
            num_local_redundant=num_local_redundant,
            distribution=args.scaling_distribution,
        )
        cpu_total = results["CPU"]["total"]["mean"] * 1e6
        gpu_total = results["GPU"]["total"]["mean"] * 1e6
        rows.append(
            (
                num_local_master,
                num_local_redundant,
                cpu_total,
                gpu_total,
                cpu_total / gpu_total if gpu_total > 0 else 0.0,
            )
        )

    if dist.get_rank() == 0:
        print_rank0(
            f"distribution={args.scaling_distribution}, num_tokens={args.num_tokens}, topk={args.topk}"
        )
        print_rank0(
            "  num_local_master  num_local_redundant  CPU_total(us)  GPU_total(us)  speedup"
        )
        for num_local_master, num_local_redundant, cpu_total, gpu_total, ratio in rows:
            print_rank0(
                f"  {num_local_master:16d}  {num_local_redundant:19d}  "
                f"{cpu_total:13.1f}  {gpu_total:13.1f}  {ratio:7.2f}x"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPU solver vs CPU solver using the Manager API"
    )
    parser.add_argument(
        "--benchmark",
        choices=("all", "placement", "pipeline", "scaling"),
        default="all",
    )
    parser.add_argument(
        "--distributions",
        type=str,
        default="uniform,skewed",
        help="Comma-separated load distributions for placement/pipeline benchmarks",
    )
    parser.add_argument("--num-local-master-experts", type=int, default=4)
    parser.add_argument("--num-local-redundant-experts", type=int, default=2)
    parser.add_argument("--expert-fc1-numel", type=int, default=3072 * 4096)
    parser.add_argument("--expert-fc2-numel", type=int, default=1536 * 4096)
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--grad-reduce-mode", choices=("low_sm", "high_sm"), default="low_sm")
    parser.add_argument("--hot-expert-ratio-per-nvl-domain", type=float, default=0.03)
    parser.add_argument("--scaling-distribution", choices=("uniform", "skewed"), default="skewed")
    parser.add_argument("--scaling-num-local-master", type=str, default="4,8,16")
    parser.add_argument("--scaling-num-local-redundant", type=str, default="2,4,8")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    world_size = dist.get_world_size()
    distributions = parse_csv_list(args.distributions)
    for distribution in distributions:
        validate_distribution(distribution)
    print_rank0(
        f"Config: world_size={world_size}, num_tokens={args.num_tokens}, topk={args.topk}, "
        f"master={args.num_local_master_experts}, redundant={args.num_local_redundant_experts}"
    )

    try:
        if args.benchmark in ("all", "placement"):
            bench_placement_gpu_vs_cpu(args, distributions)
        if args.benchmark in ("all", "pipeline"):
            bench_full_pipeline_gpu_vs_cpu(args, distributions)
        if args.benchmark in ("all", "scaling"):
            bench_scaling(args)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
