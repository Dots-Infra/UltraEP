"""Distributed reroute integration tests for the current UltraEP runtime."""

import argparse
import os
import sys

import torch
import torch.distributed as dist

try:
    import ultra_ep
except ImportError:
    print("ERROR: Cannot import ultra_ep.", file=sys.stderr)
    sys.exit(1)


def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return dist.group.WORLD


def print_rank0(msg: str):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def create_manager(group, num_layers, num_local_master, num_local_redundant, legacy_placement=False):
    return ultra_ep.Manager(
        group=group,
        num_layers=num_layers,
        num_local_master_experts=num_local_master,
        num_local_redundant_experts=num_local_redundant,
        expert_fc1_numel=64,
        expert_fc2_numel=64,
        explicitly_destroy=True,
        legacy_placement=legacy_placement,
    )


def generate_routing_map(num_tokens, num_experts, topk, device="cuda"):
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device=device)
    for token_idx in range(num_tokens):
        expert_ids = torch.randperm(num_experts, device="cpu")[:topk]
        routing_map[token_idx, expert_ids] = True
    return routing_map


def routing_map_to_topk_ids(routing_map, topk):
    token_and_expert = routing_map.nonzero(as_tuple=False)
    assert token_and_expert.size(0) == routing_map.size(0) * topk
    return token_and_expert[:, 1].reshape(routing_map.size(0), topk).to(torch.int64)


def update_placement(manager, layer_id, num_tokens, topk, seed):
    torch.manual_seed(seed + dist.get_rank())
    routing_map = generate_routing_map(
        num_tokens, manager.num_global_logical_experts, topk
    )
    manager.update_placement(layer_id, routing_map, verify_reduced_loads=True)


def test_dense_reroute(manager, layer_id, num_tokens, topk, seed):
    torch.manual_seed(seed + dist.get_rank())
    routing_map = generate_routing_map(
        num_tokens, manager.num_global_logical_experts, topk
    )
    probs = torch.randn(
        num_tokens,
        manager.num_global_logical_experts,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )

    expanded_probs, expanded_routing_map = manager.reroute(layer_id, probs, routing_map)
    assert expanded_probs.shape == (
        num_tokens,
        manager.num_global_physical_experts,
    )
    assert expanded_routing_map.shape == (
        num_tokens,
        manager.num_global_physical_experts,
    )
    assert torch.equal(
        expanded_routing_map.sum(dim=1),
        torch.full((num_tokens,), topk, dtype=torch.int64, device="cuda"),
    )
    assert expanded_routing_map.sum().item() == routing_map.sum().item()

    expanded_probs.sum().backward()
    assert probs.grad is not None
    assert torch.equal(probs.grad != 0, routing_map)
    assert torch.allclose(
        probs.grad[routing_map],
        torch.ones_like(probs.grad[routing_map]),
    )


def test_sparse_reroute(manager, layer_id, num_tokens, topk, seed):
    torch.manual_seed(seed + dist.get_rank())
    routing_map = generate_routing_map(
        num_tokens, manager.num_global_logical_experts, topk
    )
    topk_ids = routing_map_to_topk_ids(routing_map, topk)
    manager.reroute_sparse(layer_id, topk_ids)
    assert topk_ids.shape == (num_tokens, topk)
    assert topk_ids.dtype == torch.int64
    assert int(topk_ids.min().item()) >= 0
    assert int(topk_ids.max().item()) < manager.num_global_physical_experts


def run_mode(name, legacy_placement, args):
    group = dist.group.WORLD
    manager = create_manager(
        group,
        num_layers=2,
        num_local_master=args.num_local_master,
        num_local_redundant=args.num_local_redundant,
        legacy_placement=legacy_placement,
    )
    layer_id = 0
    update_placement(manager, layer_id, args.num_tokens, args.topk, args.seed)
    test_dense_reroute(manager, layer_id, args.num_tokens, args.topk, args.seed + 17)
    test_sparse_reroute(manager, layer_id, args.num_tokens, args.topk, args.seed + 31)
    manager.destroy()
    dist.barrier()
    print_rank0(f"{name}: PASS")


def main():
    parser = argparse.ArgumentParser(description="UltraEP reroute integration tests")
    parser.add_argument("--num-local-master", type=int, default=4)
    parser.add_argument("--num-local-redundant", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    setup_distributed()
    run_mode("quota-reroute", legacy_placement=False, args=args)
    run_mode("round-robin-reroute", legacy_placement=True, args=args)
    print_rank0("All reroute tests PASS")


if __name__ == "__main__":
    main()
