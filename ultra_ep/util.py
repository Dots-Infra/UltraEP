import torch
from typing import Optional
import random
import argparse


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


def setup_placement_random(
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
    """Initialize the placement maps randomly on CPU
    Two modes: uniform vs skewed
    """
    if num_nvl_ranks is None:
        # assert all global ranks inside NVL domain
        num_nvl_ranks = num_ranks

    # Reset maps
    physical_to_logical_map.fill_(-1)
    logical_to_physical_map.fill_(-1)
    logical_replica_counts.fill_(0)

    for _layer_id, (p2l_map, l2p_map, lcnts) in enumerate(
        zip(physical_to_logical_map, logical_to_physical_map, logical_replica_counts)
    ):
        # 1. Assign masters
        # Logical expert l's master is on rank l // num_local_master
        num_local_physical = num_local_master + num_local_redundant
        num_global_logical = num_local_master * num_ranks
        for l in range(num_global_logical):
            rank = l // num_local_master
            local_idx = l % num_local_master
            p = rank * num_local_physical + local_idx

            p2l_map[p] = l
            l2p_map[l, 0] = p
            lcnts[l] = 1

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
                    0,
                    len(available_logical_experts),
                    (num_local_redundant,),
                    generator=g,
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
                target_logical_indices = [
                    available_hot_cold_experts[i] for i in indices
                ]
            else:
                raise ValueError(
                    f"Unknown replica distribution: {replica_distribution}"
                )

            # Assign these replicas to the redundant slots of rank r
            for i, l in enumerate(target_logical_indices):
                p = r * num_local_physical + num_local_master + i

                count = lcnts[l].item()
                if count < num_ranks:  # Max replicas is num_ranks
                    l2p_map[l, count] = p
                    p2l_map[p] = l
                    lcnts[l] += 1


def pretty_print_log2phy_map(tensor):
    # tensor: 2D, shape (num_rows, num_cols) or (rows, cols)
    for i in range(tensor.shape[0]):
        row = tensor[i]
        # Select positive (none -1) values:
        pos_values = [str(int(x.item())) for x in row if x.item() >= 0]
        print(
            f"Logical expert {i}: [{', '.join(pos_values)}] (count={len(pos_values)})"
        )


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

    # Test with single layer
    phy2log_map = torch.zeros(
        1, num_ranks * (num_local_master + num_local_redundant), dtype=torch.int32
    )
    log2phy_map = torch.zeros(
        1, (num_local_master * num_ranks, num_ranks), dtype=torch.int32
    )
    log_cnts = torch.zeros(1, num_local_master * num_ranks, dtype=torch.int32)

    setup_placement_random(
        num_ranks,
        num_local_master,
        num_local_redundant,
        phy2log_map,
        log2phy_map,
        log_cnts,
        seed=seed,
    )
    print(f"Uniform placement, world size = {num_ranks}, NVL domain size = {num_ranks}")
    print(phy2log_map[0])
    pretty_print_log2phy_map(log2phy_map[0])
    print(log_cnts[0])
    print("-" * 100)

    setup_placement_random(
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
    print(phy2log_map[0])
    pretty_print_log2phy_map(log2phy_map[0])
    print(log_cnts[0])
    print("-" * 100)

    setup_placement_random(
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
    print(phy2log_map[0])
    pretty_print_log2phy_map(log2phy_map[0])
    print(log_cnts[0])
    print("-" * 100)

    setup_placement_random(
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
    print(phy2log_map[0])
    pretty_print_log2phy_map(log2phy_map[0])
    print(log_cnts[0])
