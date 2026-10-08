"""CPU-only setup regressions; no CUDA extension or PyTorch install required.

Run: python -m unittest discover -s tests -p test_symmetric_runtime_setup.py -v
The allocator double reproduces Torch's separate default/available registries.
GPU allocation, synchronization and placement are covered by the distributed
test_symmetric_memory.py and test_buffer_sync.py regressions.
"""

import importlib.util
from contextlib import ExitStack
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "ultra_ep"


class Group:
    def __init__(self, ranks, rank=0):
        self.ranks = ranks
        self.local_rank = rank

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.local_rank


class RuntimeSetupTest(unittest.TestCase):
    def setUp(self):
        # The default CUDA allocator is active but absent from set_backend's
        # candidates, as in affected Torch versions. Selecting it again fails.
        self.backend = "CUDA"
        self.candidates = {"NCCL"}
        self.allocator_in_use = False
        symm = ModuleType("torch.distributed._symmetric_memory")
        symm.get_backend = Mock(side_effect=lambda device: self.backend)
        symm.set_backend = Mock(side_effect=self.select)
        self.symm = symm
        self.dist = ModuleType("torch.distributed")
        self.dist.ProcessGroup = Group
        self.dist._symmetric_memory = symm
        self.dist.get_process_group_ranks = Mock(side_effect=lambda group: group.ranks)
        self.dist.new_group = Mock(side_effect=lambda **kwargs: Group(kwargs["ranks"]))
        torch = ModuleType("torch")
        torch.distributed = self.dist
        torch.device = Mock(side_effect=lambda kind, index: (kind, index))
        torch.cuda = Mock()
        torch.cuda.current_device.return_value = 3
        self.torch = torch
        self.native = ModuleType("ultra_ep._C")
        self.native.is_runtime_initialized = Mock(return_value=False)
        self.native.init_runtime = Mock(side_effect=self.initialize_native)
        self.native.destroy_runtime = Mock(side_effect=self.destroy_native)
        self.native.IpcManager = Mock()
        self.native.IpcManager.return_value.detect_accessible_ranks.return_value = 4
        package = ModuleType("ultra_ep")
        package.__path__ = [str(SOURCE)]
        package._C = self.native
        util = ModuleType("ultra_ep.util")
        util.print_rank_0 = Mock()
        modules = {
            "torch": torch,
            "torch.distributed": self.dist,
            "torch.distributed._symmetric_memory": symm,
            "ultra_ep": package,
            "ultra_ep._C": self.native,
            "ultra_ep.util": util,
        }
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.memory = self.load_module("symmetric_memory")
        self.runtime = self.load_module("runtime")

    def load_module(self, name):
        fullname = f"ultra_ep.{name}"
        spec = importlib.util.spec_from_file_location(fullname, SOURCE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        return module

    def select(self, name):
        if name not in self.candidates:
            raise RuntimeError(f"No allocation backend {name}")
        if self.allocator_in_use:
            raise RuntimeError("Backend can not be changed after use")
        self.backend = name

    def initialize_native(self, *args):
        self.native.is_runtime_initialized.return_value = True

    def destroy_native(self):
        self.native.is_runtime_initialized.return_value = False

    def test_default_cuda_is_preserved_without_reselecting_it(self):
        self.assertEqual(self.memory.select_backend(), "CUDA")
        self.assertEqual(self.memory.select_backend(), "CUDA")
        self.symm.set_backend.assert_not_called()

    def test_explicit_nccl_switches_only_once(self):
        os.environ["ULTRA_EP_SYMMETRIC_MEMORY_BACKEND"] = " nccl "
        self.assertEqual(self.memory.select_backend(), "NCCL")
        self.assertEqual(self.memory.select_backend(), "NCCL")
        self.symm.set_backend.assert_called_once_with("NCCL")

    def test_already_active_nccl_is_not_reselected(self):
        self.backend, self.candidates = "NCCL", {"CUDA"}
        os.environ["ULTRA_EP_SYMMETRIC_MEMORY_BACKEND"] = "NCCL"
        self.assertEqual(self.memory.select_backend(), "NCCL")
        self.symm.set_backend.assert_not_called()

    def test_default_moves_away_from_nvshmem(self):
        self.backend, self.candidates = "NVSHMEM", {"CUDA", "NCCL"}
        self.assertEqual(self.memory.select_backend(), "CUDA")
        self.symm.set_backend.assert_called_once_with("CUDA")

    def test_cuda_unavailable_does_not_silently_select_nccl(self):
        self.backend = None
        with self.assertRaisesRegex(RuntimeError, "No allocation backend CUDA"):
            self.memory.select_backend()
        self.symm.set_backend.assert_called_once_with("CUDA")
        self.assertIsNone(self.backend)

    def test_in_use_backend_error_is_preserved(self):
        self.backend, self.candidates = "NCCL", {"CUDA"}
        self.allocator_in_use = True
        with self.assertRaisesRegex(RuntimeError, "changed after use"):
            self.memory.select_backend()
        self.assertEqual(self.backend, "NCCL")

    def test_nvshmem_cannot_be_requested(self):
        os.environ["ULTRA_EP_SYMMETRIC_MEMORY_BACKEND"] = "NVSHMEM"
        with self.assertRaisesRegex(ValueError, "CUDA or NCCL"):
            self.memory.select_backend()
        self.symm.set_backend.assert_not_called()

    def test_single_cuda_domain_reuses_ep_group(self):
        group = Group([1, 3, 5, 7], rank=2)
        self.assertEqual(self.runtime.init_runtime(group), 4)
        self.assertIs(self.runtime.get_symmetric_group(), group)
        self.dist.new_group.assert_not_called()
        self.torch.cuda.current_device.assert_not_called()

    def test_multiple_domains_keep_global_rank_mapping_without_eager_comm(self):
        os.environ["MAX_NUM_NVL_PEERS"] = "2"
        group = Group([1, 3, 5, 7], rank=3)
        self.runtime.init_runtime(group)
        self.dist.new_group.assert_called_once_with(
            ranks=[5, 7],
            backend="nccl",
            use_local_synchronization=True,
            device_id=None,
        )
        self.native.init_runtime.assert_called_once_with(3, 4, 2)

    def test_explicit_nccl_keeps_eager_domain_group(self):
        os.environ["ULTRA_EP_SYMMETRIC_MEMORY_BACKEND"] = "NCCL"
        group = Group([1, 3, 5, 7])
        self.runtime.init_runtime(group)
        self.dist.new_group.assert_called_once_with(
            ranks=group.ranks,
            backend="nccl",
            use_local_synchronization=True,
            device_id=("cuda", 3),
        )

    def test_subgroup_is_reused_after_last_manager_is_destroyed(self):
        os.environ["MAX_NUM_NVL_PEERS"] = "2"
        group = Group([0, 1, 2, 3])
        self.runtime.init_runtime(group)
        subgroup = self.runtime.get_symmetric_group()
        self.runtime.retain_runtime()
        self.runtime.retain_runtime()
        self.runtime.release_runtime()
        self.native.destroy_runtime.assert_not_called()
        self.runtime.release_runtime()
        self.native.destroy_runtime.assert_called_once()
        self.runtime.init_runtime(group)
        self.assertIs(self.runtime.get_symmetric_group(), subgroup)
        self.dist.new_group.assert_called_once()

    def test_invalid_domain_size_does_not_create_a_group(self):
        os.environ["MAX_NUM_NVL_PEERS"] = "3"
        with self.assertRaisesRegex(ValueError, "must divide"):
            self.runtime.init_runtime(Group([0, 1, 2, 3]))
        self.dist.new_group.assert_not_called()
        self.native.init_runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
