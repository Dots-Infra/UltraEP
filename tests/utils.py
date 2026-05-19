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


SECTION_WIDTH = 96


def emit_line(msg: str = "", print_fn=None):
    if print_fn is None:
        print(msg, flush=True)
    else:
        print_fn(msg)


def print_section(title: str, sep: str = "=", print_fn=None):
    emit_line("", print_fn)
    emit_line(sep * SECTION_WIDTH, print_fn)
    emit_line(title, print_fn)
    emit_line(sep * SECTION_WIDTH, print_fn)


def print_metric(
    name: str, e2e_ms: float, kernel_ms: float, extra: str = "", print_fn=None
):
    suffix = f" | {extra}" if extra else ""
    emit_line(
        f"  {name:<28} e2e {e2e_ms:>9.3f} ms | kernel {kernel_ms:>9.3f} ms{suffix}",
        print_fn,
    )


def format_load_imbalance(summary: dict) -> str:
    return (
        f"(Constructed: rank max/mean = {summary['rank']:.3f} | "
        f"expert max/mean = {summary['expert']:.3f})"
    )


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

    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )
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

    def max_reduce_duration(value):
        if not dist.is_initialized():
            return value
        if isinstance(value, list):
            if not value:
                return value
            tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            return tensor.cpu().tolist()
        tensor = torch.tensor([float(value)], dtype=torch.float64, device="cuda")
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

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

    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return per-call total kernel durations. A logical operation may launch the
    # same kernel multiple times (e.g. relay weight sync stage 1 + stage 2), so
    # divide by fn invocations rather than by kernel launch count.
    kernel_durations = []
    events = profiler.key_averages()
    for name in kernel_names:
        total_time_us = 0.0
        for event in events:
            if name not in str(event.key):
                continue
            device_time_us = getattr(
                event, "cuda_time_total", getattr(event, "device_time_total", 0.0)
            )
            total_time_us += float(device_time_us)
        kernel_durations.append(total_time_us / 1e6 / max(num_tests, 1))

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

    # Communication tests are limited by the slowest rank. Report that rank's
    # kernel duration rather than the local rank duration.
    kernel_durations = [max_reduce_duration(duration) for duration in kernel_durations]

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

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
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
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    )


def rank_token_count(rank: int, max_tokens: int, variable: bool, seed: int) -> int:
    if not variable:
        return max_tokens
    rng = random.Random(seed + 7919 * rank)
    sample = rng.gauss(0.5, 0.35)
    return max(0, min(max_tokens, int(round(sample * max_tokens))))


def topk_inclusion_probs_approx(weights: torch.Tensor, topk: int) -> torch.Tensor:
    """Approximate weighted top-k inclusion probabilities without replacement."""
    if topk <= 0:
        return torch.zeros_like(weights)
    if topk >= weights.numel():
        return torch.ones_like(weights)

    weights = weights.to(torch.float64)
    lo, hi = 0.0, 1.0
    while float((1.0 - torch.exp(-hi * weights)).sum().item()) < topk:
        hi *= 2.0
    for _ in range(48):
        mid = (lo + hi) * 0.5
        expected = float((1.0 - torch.exp(-mid * weights)).sum().item())
        if expected < topk:
            lo = mid
        else:
            hi = mid
    return 1.0 - torch.exp(-hi * weights)


def expert_weights_for_rank_alpha(
    num_experts: int,
    num_ranks: int,
    num_local_master: int,
    rank_alpha: float,
    local_expert_alpha: float,
) -> torch.Tensor:
    rank_ids = torch.arange(1, num_ranks + 1, dtype=torch.float64)
    rank_weights = 1.0 / rank_ids.pow(rank_alpha)
    local_ids = torch.arange(1, num_local_master + 1, dtype=torch.float64)
    local_weights = 1.0 / local_ids.pow(local_expert_alpha)
    weights = (rank_weights.unsqueeze(1) * local_weights.unsqueeze(0)).reshape(
        num_experts
    )
    return (weights / weights.sum()).to(torch.float32)


def zipf_alpha_for_rank_ratio(
    num_experts: int,
    num_ranks: int,
    num_local_master: int,
    topk: int,
    target_ratio: float,
) -> float:
    if target_ratio < 1.0:
        raise ValueError("imbalance_ratio must be >= 1")
    if target_ratio == 1.0:
        return 0.0

    local_expert_alpha = 4.0 if num_local_master > 1 else 0.0

    def ratio_for_alpha(alpha: float) -> float:
        weights = expert_weights_for_rank_alpha(
            num_experts,
            num_ranks,
            num_local_master,
            alpha,
            local_expert_alpha,
        )
        # The real routing uses top-k without replacement, so raw probability
        # mass overstates hot-expert load once inclusion probabilities saturate.
        rank_loads = (
            topk_inclusion_probs_approx(weights, topk=topk)
            .view(num_ranks, num_local_master)
            .sum(dim=1)
        )
        return (rank_loads.max() / rank_loads.mean()).item()

    lo, hi = 0.0, 8.0
    for _ in range(32):
        mid = (lo + hi) * 0.5
        if ratio_for_alpha(mid) < target_ratio:
            lo = mid
        else:
            hi = mid
    return hi


def zipf_weights_for_ratio(
    num_experts: int,
    num_ranks: int,
    num_local_master: int,
    topk: int,
    imbalance_ratio: float,
) -> torch.Tensor:
    if imbalance_ratio < 1.0:
        raise ValueError("imbalance_ratio must be >= 1")
    if imbalance_ratio == 1.0:
        return torch.ones(num_experts, dtype=torch.float32) / num_experts
    alpha = zipf_alpha_for_rank_ratio(
        num_experts, num_ranks, num_local_master, topk, imbalance_ratio
    )
    local_expert_alpha = 4.0 if num_local_master > 1 else 0.0
    return expert_weights_for_rank_alpha(
        num_experts, num_ranks, num_local_master, alpha, local_expert_alpha
    )


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
    if imbalance_ratio < 1.0:
        raise ValueError("imbalance_ratio must be >= 1")
    if imbalance_ratio == 1.0:
        token_ids = torch.arange(num_tokens, device=device).unsqueeze(1)
        offsets = torch.arange(topk, device=device).unsqueeze(0)
        return ((token_ids * topk + offsets + rank * topk) % num_experts).to(
            torch.int64
        )

    weights = zipf_weights_for_ratio(
        num_experts, num_ranks, num_local_master, topk, imbalance_ratio
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
    variable_input_tokens: bool,
    imbalance_ratio: float,
    seed: int,
    device: str = "cuda",
) -> torch.Tensor:
    rows = []
    for rank in range(num_ranks):
        ntokens = rank_token_count(rank, tokens_per_rank, variable_input_tokens, seed)
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
        rows.append(
            torch.bincount(topk_ids.flatten(), minlength=num_experts).to(torch.int32)
        )
    return torch.stack(rows, dim=0)


def load_imbalance_summary(
    loads_per_rank: torch.Tensor, num_ranks: int, num_local_master: int
):
    expert_loads = loads_per_rank.sum(dim=0, dtype=torch.int32)
    return expert_load_imbalance_summary(expert_loads, num_ranks, num_local_master)


def expert_load_imbalance_summary(
    expert_loads: torch.Tensor, num_ranks: int, num_local_master: int
):
    rank_loads = expert_loads.view(num_ranks, num_local_master).sum(dim=1)
    return {
        "rank": max_mean(rank_loads).item(),
        "expert": max_mean(expert_loads).item(),
    }


def max_mean(t: torch.Tensor) -> torch.Tensor:
    values = t.float()
    mean = values.mean()
    return torch.where(mean > 0, values.max() / mean, torch.ones_like(mean))


def nvl_domain_physical_lower_bound(
    rank_loads: torch.Tensor, nvl_domain_size: int
) -> torch.Tensor:
    values = rank_loads.float()
    mean = values.mean()
    if nvl_domain_size <= 0 or values.numel() % nvl_domain_size != 0:
        raise ValueError("nvl_domain_size must evenly divide rank_loads")
    domain_means = values.view(-1, nvl_domain_size).mean(dim=1)
    return torch.where(mean > 0, domain_means.max() / mean, torch.ones_like(mean))


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
