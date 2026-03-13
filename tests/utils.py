import torch
import torch.distributed as dist
from typing import Callable, Optional, Union
import os
import sys
import json
from pathlib import Path
import tempfile
import warnings
import numpy as np
import random


class suppress_stdout_stderr:

    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


class empty_suppress:

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# Disable all PyTorch profiler warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.profiler")


def build_logical_expert_weights(
    distribution: str,
    num_global_logical_experts: int,
    seed: int,
    num_ranks: Optional[int] = None,
    num_local_master: Optional[int] = None,
    num_nvl_ranks: Optional[int] = None,
    hot_expert_ratio_per_nvl_domain: float = 0.03,
    zipf_alpha: float = 1.2,
    single_hot_ratio: float = 0.8,
):
    if distribution == "uniform":
        weights = torch.ones(num_global_logical_experts, dtype=torch.float32)
    elif distribution == "zipf":
        ranks = torch.arange(1, num_global_logical_experts + 1, dtype=torch.float32)
        weights = 1.0 / ranks.pow(zipf_alpha)
    elif distribution == "single_hot":
        weights = torch.full(
            (num_global_logical_experts,),
            (1.0 - single_hot_ratio) / max(num_global_logical_experts - 1, 1),
            dtype=torch.float32,
        )
        weights[0] = single_hot_ratio
    elif distribution == "skewed":
        assert num_ranks is not None
        assert num_local_master is not None
        if num_nvl_ranks is None:
            num_nvl_ranks = num_ranks

        weights = torch.zeros(num_global_logical_experts, dtype=torch.float32)
        rng = random.Random(seed)
        num_nvl_domains = (num_ranks + num_nvl_ranks - 1) // num_nvl_ranks

        for nvl_domain in range(num_nvl_domains):
            domain_start_rank = nvl_domain * num_nvl_ranks
            domain_end_rank = min((nvl_domain + 1) * num_nvl_ranks, num_ranks)
            logical_experts = []
            for rank in range(domain_start_rank, domain_end_rank):
                logical_experts.extend(
                    range(rank * num_local_master, (rank + 1) * num_local_master)
                )

            if not logical_experts:
                continue

            num_hot = max(
                1,
                int(len(logical_experts) * hot_expert_ratio_per_nvl_domain),
            )
            hot_experts = set(rng.sample(logical_experts, num_hot))
            cold_experts = [idx for idx in logical_experts if idx not in hot_experts]

            hot_weight = 0.9 / len(hot_experts) if hot_experts else 0.0
            cold_weight = 0.1 / len(cold_experts) if cold_experts else 0.0

            for logical_idx in hot_experts:
                weights[logical_idx] = hot_weight
            for logical_idx in cold_experts:
                weights[logical_idx] = cold_weight
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return weights / weights.sum()


def generate_routing_map_from_distribution(
    num_tokens: int,
    num_global_logical_experts: int,
    topk: int,
    distribution: str,
    seed: int,
    device: str = "cuda",
    num_ranks: Optional[int] = None,
    num_local_master: Optional[int] = None,
    num_nvl_ranks: Optional[int] = None,
    hot_expert_ratio_per_nvl_domain: float = 0.03,
    zipf_alpha: float = 1.2,
    single_hot_ratio: float = 0.8,
):
    weights = build_logical_expert_weights(
        distribution=distribution,
        num_global_logical_experts=num_global_logical_experts,
        seed=seed,
        num_ranks=num_ranks,
        num_local_master=num_local_master,
        num_nvl_ranks=num_nvl_ranks,
        hot_expert_ratio_per_nvl_domain=hot_expert_ratio_per_nvl_domain,
        zipf_alpha=zipf_alpha,
        single_hot_ratio=single_hot_ratio,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    weight_matrix = weights.unsqueeze(0).repeat(num_tokens, 1)
    topk_ids = torch.multinomial(
        weight_matrix,
        num_samples=topk,
        replacement=False,
        generator=generator,
    )

    routing_map = torch.zeros(
        num_tokens,
        num_global_logical_experts,
        dtype=torch.bool,
        device=device,
    )
    token_ids = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, topk)
    routing_map[token_ids, topk_ids.to(device)] = True
    return routing_map


def bench(
    fn,
    num_warmups: int = 50,
    num_tests: int = 50,
    use_barrier: bool = True,
    pre_fn=None,
    post_fn=None,
):
    stats = bench_stats(
        fn,
        num_warmups=num_warmups,
        num_tests=num_tests,
        use_barrier=use_barrier,
        pre_fn=pre_fn,
        post_fn=post_fn,
    )
    return stats["mean"], stats["min"], stats["max"]


def bench_stats(
    fn,
    num_warmups: int = 50,
    num_tests: int = 50,
    use_barrier: bool = True,
    pre_fn=None,
    post_fn=None,
):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        if pre_fn is not None:
            pre_fn()
        if use_barrier and dist.is_initialized():
            dist.barrier()
        start_events[i].record()
        fn()
        if use_barrier and dist.is_initialized():
            dist.barrier()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]
    if times.size >= 3:
        times = times[np.argsort(times)][1:-1]

    return {
        "mean": float(np.mean(times)) if times.size > 0 else 0.0,
        "min": float(np.min(times)) if times.size > 0 else 0.0,
        "max": float(np.max(times)) if times.size > 0 else 0.0,
        "p50": float(np.percentile(times, 50)) if times.size > 0 else 0.0,
        "p99": float(np.percentile(times, 99)) if times.size > 0 else 0.0,
        "num_samples": int(times.size),
    }


def bench_kineto(
    fn,
    kernel_names: Union[str, tuple],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    barrier_comm_profiling: bool = False,
    num_kernels_per_period: int = 1,
    barrier: Optional[Callable] = None,
):
    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)

    # Skip profiling
    # Conflict with Nsight Systems, Nsight Compute and Compute Sanitizer
    if int(os.environ.get("EP_USE_NVIDIA_TOOLS", 0)):
        return (1,) * len(kernel_names) if is_tuple else 1

    # For some auto-tuning kernels with prints
    fn()
    torch.cuda.synchronize()

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
        )
        with profiler:
            for i in range(2):
                for _ in range(num_tests):
                    # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                    if barrier_comm_profiling:
                        lhs = torch.randn(
                            (8192, 8192), dtype=torch.float, device="cuda"
                        )
                        rhs = torch.randn(
                            (8192, 8192), dtype=torch.float, device="cuda"
                        )
                        lhs @ rhs

                        # Some network may have ring-based implement, so be careful to use `all_reduce`
                        if barrier is None:
                            dist.all_reduce(
                                torch.ones(1, dtype=torch.float, device="cuda")
                            )
                        else:
                            barrier()
                    fn()
                torch.cuda.synchronize()
                profiler.step()

    # Parse the profiling table
    prof_lines = (
        profiler.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert (
            sum([name in line for line in prof_lines]) <= 1
        ), f"Errors of the kernel {name} in the profiling table: {prof_lines}"

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel durations
    units = {"ms": 1e3, "us": 1e6}
    kernel_durations = []
    for name in kernel_names:
        total_time = 0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += (
                            float(time_str.replace(unit, "")) / scale * int(num_str)
                        )
                        total_num += int(num_str)
                        break
        kernel_durations.append(total_time / total_num if total_num > 0 else 0)

    # Expand the kernels by periods
    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            profiler.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())

        for i, kernel_name in enumerate(kernel_names):
            events = [
                event
                for event in profile_data["traceEvents"]
                if f"::{kernel_name}" in event["name"]
            ]
            events = sorted(events, key=lambda event: event["ts"])
            durations = [event["dur"] / 1e6 for event in events]
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations[i] = [
                sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                for j in range(num_kernels_per_period)
            ]

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]
