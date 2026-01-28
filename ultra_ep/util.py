import torch


def print_rank_0(message):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(message)
