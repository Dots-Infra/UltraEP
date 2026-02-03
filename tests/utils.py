import torch
import torch.distributed as dist
import random
from typing import Callable, DefaultDict, Optional, Union
import argparse
import os
import sys
import json
from pathlib import Path
import tempfile
import warnings
import numpy as np


def setup_placement(
    num_ranks: int,
    num_local_master: int,
    num_local_redundant: int,
    physical_to_logical_map: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_counts: torch.Tensor,
    replica_distribution: str = "uniform",
    num_nvl_ranks: Optional[int] = None,
    hot_expert_ratio_per_nvl_domain: float = 0.04,
    seed: int = 42,
):
    if num_nvl_ranks is None:
        # assert all global ranks inside NVL domain
        num_nvl_ranks = num_ranks

    # Reset maps
    physical_to_logical_map.fill_(-1)
    logical_to_physical_map.fill_(-1)
    logical_replica_counts.fill_(0)

    # 1. Assign masters
    # Logical expert l's master is on rank l // num_local_master
    num_local_physical = num_local_master + num_local_redundant
    num_global_logical = num_local_master * num_ranks
    for l in range(num_global_logical):
        rank = l // num_local_master
        local_idx = l % num_local_master
        p = rank * num_local_physical + local_idx

        physical_to_logical_map[p] = l
        logical_to_physical_map[l, 0] = p
        logical_replica_counts[l] = 1

    # 2. Assign redundant experts (replicas)
    # Redundant slots on rank r can only serve as replicas for logical experts
    # whose master is in the same NVL domain as rank r.

    # Group logical experts by NVL domain of their master rank
    logical_experts_by_nvl_domain = [
        [] for _ in range((num_ranks + num_nvl_ranks - 1) // num_nvl_ranks)
    ]
    for l in range(num_global_logical):
        master_rank = l // num_local_master
        nvl_domain = master_rank // num_nvl_ranks
        logical_experts_by_nvl_domain[nvl_domain].append(l)

    num_logical_per_nvl_domain = num_local_master * num_nvl_ranks
    num_hot_logical_per_nvl_domain = max(
        1, int(num_logical_per_nvl_domain * hot_expert_ratio_per_nvl_domain)
    )
    hot_experts_each_nvl_domain = {}

    # For each rank, assign its redundant slots
    g = torch.Generator()
    g.manual_seed(seed)  # Consistent across ranks
    random.seed(seed)

    for r in range(num_ranks):
        nvl_domain = r // num_nvl_ranks

        # Predetermine hot experts in this nvl domain
        if nvl_domain not in hot_experts_each_nvl_domain:
            hot_experts_each_nvl_domain[nvl_domain] = random.sample(
                logical_experts_by_nvl_domain[nvl_domain],
                num_hot_logical_per_nvl_domain,
            )

        # Filter out logical experts whose master is on this rank
        # (a replica cannot be on the same rank as its master)
        master_logical_experts_on_this_rank = set(
            range(r * num_local_master, (r + 1) * num_local_master)
        )
        available_logical_experts = [
            l
            for l in logical_experts_by_nvl_domain[nvl_domain]
            if l not in master_logical_experts_on_this_rank
        ]

        if not available_logical_experts:
            continue

        available_hot_experts = [
            l
            for l in hot_experts_each_nvl_domain[nvl_domain]
            if l not in master_logical_experts_on_this_rank
        ]
        available_cold_experts = [
            l for l in available_logical_experts if l not in available_hot_experts
        ]
        available_hot_cold_experts = available_hot_experts + available_cold_experts
        num_hot = len(available_hot_experts)
        num_cold = len(available_cold_experts)
        hot_expert_weights = [0.9 / num_hot] * num_hot if num_hot > 0 else []
        cold_expert_weights = [0.1 / num_cold] * num_cold if num_cold > 0 else []
        expert_weights = hot_expert_weights + cold_expert_weights

        if replica_distribution == "uniform":
            # Pick logical experts from this domain uniformly at random
            indices = torch.randint(
                0, len(available_logical_experts), (num_local_redundant,), generator=g
            ).tolist()
            target_logical_indices = [available_logical_experts[i] for i in indices]
        elif replica_distribution == "skewed":
            # Pick a small subset of "hot" experts in this domain
            indices = torch.multinomial(
                torch.tensor(expert_weights),
                num_local_redundant,
                replacement=False,
                generator=g,
            ).tolist()
            target_logical_indices = [available_hot_cold_experts[i] for i in indices]
        else:
            raise ValueError(f"Unknown replica distribution: {replica_distribution}")

        # Assign these replicas to the redundant slots of rank r
        for i, l in enumerate(target_logical_indices):
            p = r * num_local_physical + num_local_master + i

            count = logical_replica_counts[l].item()
            if count < num_ranks:  # Max replicas is num_ranks
                logical_to_physical_map[l, count] = p
                physical_to_logical_map[p] = l
                logical_replica_counts[l] += 1


def pretty_print_log2phy_map(tensor):
    # tensor: 2D, shape (num_rows, num_cols) or (rows, cols)
    for i in range(tensor.shape[0]):
        row = tensor[i]
        # Select positive (none -1) values:
        pos_values = [str(int(x.item())) for x in row if x.item() >= 0]
        print(
            f"Logical expert {i}: [{', '.join(pos_values)}] (count={len(pos_values)})"
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
        if use_barrier:
            dist.barrier()
        start_events[i].record()
        fn()
        if use_barrier:
            dist.barrier()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]
    return np.average(times), np.min(times), np.max(times)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-ranks", type=int, default=32)
    parser.add_argument("--num-local-master", type=int, default=4)
    parser.add_argument("--num-local-redundant", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    num_ranks = args.num_ranks
    num_local_master = args.num_local_master
    num_local_redundant = args.num_local_redundant
    seed = args.seed

    phy2log_map = torch.zeros(
        num_ranks * (num_local_master + num_local_redundant), dtype=torch.int32
    )
    log2phy_map = torch.zeros(
        (num_local_master * num_ranks, num_ranks), dtype=torch.int32
    )
    log_cnts = torch.zeros(num_local_master * num_ranks, dtype=torch.int32)

    setup_placement(
        num_ranks,
        num_local_master,
        num_local_redundant,
        phy2log_map,
        log2phy_map,
        log_cnts,
        seed=seed,
    )
    print(f"Uniform placement, world size = {num_ranks}, NVL domain size = {num_ranks}")
    print(phy2log_map)
    pretty_print_log2phy_map(log2phy_map)
    print(log_cnts)
    print("-" * 100)

    setup_placement(
        num_ranks,
        num_local_master,
        num_local_redundant,
        phy2log_map,
        log2phy_map,
        log_cnts,
        replica_distribution="skewed",
        hot_expert_ratio_per_nvl_domain=0.03,
        seed=seed,
    )
    print(f"Skewed placement, world size = {num_ranks}, NVL domain size = {num_ranks}")
    print(phy2log_map)
    pretty_print_log2phy_map(log2phy_map)
    print(log_cnts)
    print("-" * 100)

    setup_placement(
        num_ranks,
        num_local_master,
        num_local_redundant,
        phy2log_map,
        log2phy_map,
        log_cnts,
        replica_distribution="uniform",
        num_nvl_ranks=8,
        seed=seed,
    )
    print(f"Uniform placement, world size = {num_ranks}, NVL domain size = 8")
    print(phy2log_map)
    pretty_print_log2phy_map(log2phy_map)
    print(log_cnts)
    print("-" * 100)

    setup_placement(
        num_ranks,
        num_local_master,
        num_local_redundant,
        phy2log_map,
        log2phy_map,
        log_cnts,
        replica_distribution="skewed",
        num_nvl_ranks=8,
        hot_expert_ratio_per_nvl_domain=0.03,
        seed=seed,
    )
    print(f"Skewed placement, world size = {num_ranks}, NVL domain size = 8")
    print(phy2log_map)
    pretty_print_log2phy_map(log2phy_map)
    print(log_cnts)
