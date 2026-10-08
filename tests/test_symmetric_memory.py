"""Distributed allocation, placement and lifetime regression.

Run with torchrun --standalone --nproc_per_node=4 tests/test_symmetric_memory.py.
Use --subgroups to exercise nonconsecutive global ranks, or set
MAX_NUM_NVL_PEERS=2 to split the EP group into two symmetric-memory domains.
"""

import argparse
import faulthandler
import gc
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import ultra_ep
from ultra_ep.symmetric_memory import allocate_region
from ultra_ep.runtime import get_symmetric_group


def check_placement(manager, graph=False):
    group = manager.group
    experts = manager.num_global_logical_experts
    tokens = 127 + group.rank() * 3
    # Both inputs are noncontiguous so the wrapper must preserve temporary
    # storage while its communication stream and NCCL consume the data.
    ids = (torch.arange(tokens * 4, device="cuda") + group.rank()) % experts
    ids = ids.reshape(tokens, 4)[:, ::2]
    dense = torch.zeros((experts, tokens), device="cuda", dtype=torch.bool).t()
    dense.scatter_(1, ids, True)
    expected = dense.sum(0, dtype=torch.int32)
    dist.all_reduce(expected, group=group)
    source_stream = torch.cuda.Stream()
    source_stream.wait_stream(torch.cuda.current_stream())

    def update():
        for layer, method, data in (
            (0, manager.update_placement, dense),
            (1, manager.update_placement_sparse, ids),
        ):
            with (
                patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce,
                patch.object(dist, "all_gather_into_tensor", wraps=dist.all_gather_into_tensor) as gather,
            ):
                method(layer, data)
            assert reduce.call_count == int(manager.legacy_placement)
            assert gather.call_count == int(not manager.legacy_placement)
            collective = reduce if manager.legacy_placement else gather
            assert collective.call_args.kwargs["group"] is group
        torch.cuda.current_stream().wait_stream(manager.get_comm_stream())

    with torch.cuda.stream(source_stream):
        update()
    torch.cuda.synchronize()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=source_stream):
            update()
        for shift in (1, 3, 5):
            ids.add_(shift).remainder_(experts)
            dense.zero_().scatter_(1, ids, True)
            expected = dense.sum(0, dtype=torch.int32)
            dist.all_reduce(expected, group=group)
            captured.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                manager.runtime.get_global_logical_expert_loads_tensor(), expected
            )
        del captured
    torch.testing.assert_close(
        manager.runtime.get_global_logical_expert_loads_tensor(), expected
    )
    torch.testing.assert_close(
        manager.physical_to_logical_map[0], manager.physical_to_logical_map[1]
    )
    copies = [
        torch.empty_like(manager.physical_to_logical_map[0])
        for _ in range(group.size())
    ]
    dist.all_gather(copies, manager.physical_to_logical_map[0], group=group)
    for other in copies:
        torch.testing.assert_close(other, copies[0])


def check_remote_views(manager):
    group = get_symmetric_group()
    for dtype, count in ((torch.uint8, 0), (torch.float32, 3), (torch.uint64, 2)):
        region = allocate_region(
            count,
            dtype=dtype,
            device=torch.device("cuda", torch.cuda.current_device()),
            group=group,
            local_shape=(count,),
        )
        assert region.local.numel() == count
        assert region.storage.numel() * region.storage.element_size() >= 16
        assert region.storage.data_ptr() % 16 == 0
        assert region.remote_ptrs.numel() == group.size()
        region.local.fill_(group.rank() + 1)
        region.handle.barrier(channel=0)
        for peer in range(group.size()):
            view = region.handle.get_buffer(peer, (count,), dtype)
            torch.testing.assert_close(view, torch.full_like(view, peer + 1))
        region.handle.barrier(channel=1)
        torch.cuda.synchronize()
        del view, region


def make_manager(group, *, replicas=1, train=True, legacy=False, explicit=True):
    return ultra_ep.Manager(
        group=group,
        num_layers=2,
        num_local_master_experts=2,
        num_local_redundant_experts=replicas,
        expert_fc1_numel=128,
        expert_fc2_numel=64,
        is_train=train,
        legacy_placement=legacy,
        explicitly_destroy=explicit,
    )


def check_empty_buffers(manager):
    weights = [
        torch.ones((2, n), device="cuda", dtype=torch.bfloat16) for n in (128, 64)
    ]
    grads = [torch.ones_like(weight, dtype=torch.float32) for weight in weights]
    manager.construct_local_master_ptr_pool(
        0, *weights, *(grads if manager.is_train else (None, None))
    )
    manager.weight_sync(0, async_finish=True).current_stream_wait()
    if manager.is_train:
        manager.grad_reduce(
            0, async_finish=True, use_current_stream=True
        ).current_stream_wait()
        for grad in grads:
            torch.testing.assert_close(grad, torch.ones_like(grad))
    assert manager.local_replica_weight_buffer.numel() == 0


def main():
    faulthandler.dump_traceback_later(45, repeat=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subgroups", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # Deliberately leave communicator initialization lazy: Manager supports it.
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    groups = []
    if args.subgroups:
        assert dist.get_world_size() >= 4
        for parity in (0, 1):
            candidate = dist.new_group(list(range(parity, dist.get_world_size(), 2)))
            if dist.get_rank() % 2 == parity:
                group = candidate
                groups.append(candidate)

    ultra_ep.init_runtime(group)  # Public pre-initialization must not acquire a manager reference.
    backend = os.getenv("ULTRA_EP_SYMMETRIC_MEMORY_BACKEND", "").strip().upper() or "CUDA"
    assert symm_mem.get_backend("cuda") == backend
    if backend == "CUDA" and get_symmetric_group().size() == group.size():
        assert get_symmetric_group() is group
    first = make_manager(group)
    second = make_manager(group)
    if group.rank() == 0:
        print("Checking peer views and shared runtime", flush=True)
    check_remote_views(first)
    check_placement(first, graph=True)
    first.destroy()
    first.destroy()  # Idempotent teardown must not release the shared runtime twice.
    check_placement(second, graph=True)
    second.destroy()

    for train in (True, False):
        if group.rank() == 0:
            print(f"Checking empty buffers, train={train}", flush=True)
        empty = make_manager(group, replicas=0, train=train)
        if group.rank() == 0:
            print("Empty manager initialized", flush=True)
        check_placement(empty)
        check_empty_buffers(empty)
        empty.destroy()
    legacy = make_manager(group, legacy=True)
    check_placement(legacy)
    legacy.destroy()
    automatic = make_manager(group, explicit=False)
    check_placement(automatic)
    del automatic
    gc.collect()
    assert not ultra_ep._C.is_runtime_initialized()
    if dist.get_rank() == 0:
        print(
            "PASS symmetric allocation, placement, graphs, empty buffers and lifetime",
            flush=True,
        )
    for subgroup in groups:
        dist.destroy_process_group(subgroup)
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
