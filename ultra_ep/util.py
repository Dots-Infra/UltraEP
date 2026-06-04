import torch


def print_rank_0(message):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(message, flush=True)


def print_rank_k(message, rank):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == rank:
        print(message, flush=True)


def get_max_by_mean(load_tensor):
    load_tensor = load_tensor.float()
    max_load = load_tensor.max().item()
    mean_load = load_tensor.mean().item()
    return max_load / mean_load if mean_load > 0 else 1.0
