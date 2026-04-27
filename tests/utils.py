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

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])
    if times.size > 1:
        times = times[1:]
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


def bench_cuda_event_groups(
    fn,
    groups: dict[str, tuple[str, ...]],
    num_tests: int = 10,
    use_barrier: bool = True,
) -> dict[str, float]:
    """Profile CUDA events and sum durations by substring groups."""
    fn()
    torch.cuda.synchronize()
    if use_barrier and dist.is_initialized():
        dist.barrier()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(num_tests):
            fn()
        torch.cuda.synchronize()

    out = {name: 0.0 for name in groups}
    for event in profiler.key_averages():
        key = event.key.lower()
        device_time_us = getattr(
            event, "cuda_time_total", getattr(event, "device_time_total", 0.0)
        )
        duration_s = float(device_time_us) / 1e6
        for group_name, needles in groups.items():
            if any(needle.lower() in key for needle in needles):
                out[group_name] += duration_s

    denom = max(num_tests, 1)
    return {name: value / denom for name, value in out.items()}


def parse_csv_strings(value: str):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def rank_token_count(rank: int, max_tokens: int, variable: bool, seed: int) -> int:
    if not variable:
        return max_tokens
    rng = random.Random(seed + 7919 * rank)
    bucket = rng.random()
    if bucket < 0.15:
        return 0
    if bucket < 0.30:
        return max_tokens
    return rng.randint(1, max_tokens - 1) if max_tokens > 1 else max_tokens


def zipf_alpha_for_rank_ratio(
    num_experts: int, num_ranks: int, num_local_master: int, target_ratio: float
) -> float:
    if target_ratio <= 1.0:
        return 0.0

    def ratio_for_alpha(alpha: float) -> float:
        ranks = torch.arange(1, num_experts + 1, dtype=torch.float64)
        weights = 1.0 / ranks.pow(alpha)
        rank_weights = weights.view(num_ranks, num_local_master).sum(dim=1)
        return (rank_weights.max() / rank_weights.mean()).item()

    lo, hi = 0.0, 8.0
    for _ in range(32):
        mid = (lo + hi) * 0.5
        if ratio_for_alpha(mid) < target_ratio:
            lo = mid
        else:
            hi = mid
    return hi


def zipf_weights_for_ratio(
    num_experts: int, num_ranks: int, num_local_master: int, imbalance_ratio: float
) -> torch.Tensor:
    if imbalance_ratio <= 0:
        return torch.ones(num_experts, dtype=torch.float32) / num_experts
    alpha = zipf_alpha_for_rank_ratio(
        num_experts, num_ranks, num_local_master, imbalance_ratio
    )
    ranks = torch.arange(1, num_experts + 1, dtype=torch.float32)
    weights = 1.0 / ranks.pow(alpha)
    return weights / weights.sum()


def generate_topk_ids_zipf(
    num_tokens: int,
    num_experts: int,
    num_ranks: int,
    num_local_master: int,
    topk: int,
    imbalance_ratio: float,
    seed: int,
    rank: int = 0,
    device: str = "cuda",
) -> torch.Tensor:
    if num_tokens == 0:
        return torch.empty((0, topk), dtype=torch.int64, device=device)
    if imbalance_ratio <= 0:
        token_ids = torch.arange(num_tokens, device=device).unsqueeze(1)
        offsets = torch.arange(topk, device=device).unsqueeze(0)
        return ((token_ids * topk + offsets + rank * topk) % num_experts).to(torch.int64)

    weights = zipf_weights_for_ratio(
        num_experts, num_ranks, num_local_master, imbalance_ratio
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + rank * 104729)
    topk_ids = torch.multinomial(
        weights.repeat(num_tokens, 1),
        num_samples=topk,
        replacement=False,
        generator=generator,
    )
    return topk_ids.to(device=device, dtype=torch.int64)


def topk_ids_to_routing_map(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    routing_map = torch.zeros(
        (topk_ids.size(0), num_experts), dtype=torch.bool, device=topk_ids.device
    )
    if topk_ids.numel() > 0:
        token_ids = torch.arange(topk_ids.size(0), device=topk_ids.device).unsqueeze(1)
        routing_map[token_ids, topk_ids] = True
    return routing_map


def generate_routing_map_zipf(
    num_tokens: int,
    num_experts: int,
    num_ranks: int,
    num_local_master: int,
    topk: int,
    imbalance_ratio: float,
    seed: int,
    rank: int = 0,
    device: str = "cuda",
) -> torch.Tensor:
    topk_ids = generate_topk_ids_zipf(
        num_tokens,
        num_experts,
        num_ranks,
        num_local_master,
        topk,
        imbalance_ratio,
        seed,
        rank=rank,
        device=device,
    )
    return topk_ids_to_routing_map(topk_ids, num_experts)


def generate_loads_per_rank_zipf(
    num_ranks: int,
    num_experts: int,
    num_local_master: int,
    topk: int,
    tokens_per_rank: int,
    variable_num_tokens: bool,
    imbalance_ratio: float,
    seed: int,
    device: str = "cuda",
) -> torch.Tensor:
    rows = []
    for rank in range(num_ranks):
        ntokens = rank_token_count(rank, tokens_per_rank, variable_num_tokens, seed)
        topk_ids = generate_topk_ids_zipf(
            ntokens,
            num_experts,
            num_ranks,
            num_local_master,
            topk,
            imbalance_ratio,
            seed,
            rank=rank,
            device=device,
        )
        rows.append(torch.bincount(topk_ids.flatten(), minlength=num_experts).to(torch.int32))
    return torch.stack(rows, dim=0)


def max_mean(t: torch.Tensor) -> torch.Tensor:
    values = t.float()
    mean = values.mean()
    return torch.where(mean > 0, values.max() / mean, torch.ones_like(mean))


def summarize_vector(t: torch.Tensor) -> dict:
    values = t.float()
    if values.numel() == 0:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": values.min().item(),
        "median": values.median().item(),
        "mean": values.mean().item(),
        "max": values.max().item(),
    }
