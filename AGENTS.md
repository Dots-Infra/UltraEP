# Repository Guidelines

## Project Structure & Module Organization
- `ultra_ep/`: Python API surface (`Manager`, runtime wrappers, reroute helpers).
- `csrc/`: CUDA/C++ extension sources.
- `csrc/kernels/`: core device kernels (`weight_sync`, `grad_reduce`, `reroute`, task build).
- `csrc/solvers/`: placement and reroute solver implementations (CPU and GPU paths).
- `csrc/utils/`: shared CUDA/NVSHMEM/runtime utilities.
- `tests/`: distributed correctness + benchmark-style tests (`test_*.py`) and helper utilities.
- `build.sh`, `setup.py`, `format.sh`: build, packaging, and formatting entry points.

## Build, Test, and Development Commands
- `./build.sh`: clean `build/`, `dist/`, `*.egg-info/` and build a wheel via `python setup.py bdist_wheel`.
- `pip install dist/*.whl`: install the locally built package.
- `python setup.py bdist_wheel`: direct build path (useful when passing env vars like `NVSHMEM_DIR=/path/to/nvshmem`).
- `./format.sh`: format C++/CUDA with `clang-format` and Python with `black` (fallback: `autopep8`).
- `torchrun --nproc_per_node=4 tests/test_weight_sync.py`: distributed kernel correctness/perf test.
- `torchrun --nproc_per_node=4 tests/test_grad_reduce.py` and `tests/test_reroute.py`: additional distributed operator validation.
- `python tests/test_placement.py --num-ranks 32 --nvl-domain-size 8 ...`: placement logic validation.

## Coding Style & Naming Conventions
- Python: `snake_case` for functions/variables/files, `PascalCase` for classes, keep module APIs under `ultra_ep/` minimal and explicit.
- C++/CUDA: follow `.clang-format` (Google base, 4-space indent, 120-column limit, left-aligned pointers/references).
- Prefer descriptive kernel/solver file names matching feature scope (for example `placement_quota_gpu.cu`).
- Run `./format.sh` before opening a PR.

## Testing Guidelines
- Test files use `tests/test_*.py`; keep new tests colocated with similar operators/solvers.
- Most tests are multi-GPU and require `torch.distributed` (`nccl`) plus NVLINK-capable hardware (SM90/SM100 expected).
- Cover both correctness and performance-sensitive paths when touching kernels/placement logic.

## Commit & Pull Request Guidelines
- Follow conventional commit prefixes used in history: `feat:`, `fix:`, `perf:`, `chore:`.
- Keep commit messages imperative and scoped (`fix: support PP/VPP with virtual layer id`).
- PRs should include:
  - what changed and why,
  - exact test commands run,
  - hardware/runtime context (GPU type, ranks, key env vars),
  - before/after metrics for performance-related changes.

## Environment & Configuration Tips
- Install NVSHMEM first (`pip install nvidia-nvshmem-cu12`) or set `NVSHMEM_DIR` explicitly.
- `setup.py` auto-detects CUDA arch via `nvidia-smi`; override with `TORCH_CUDA_ARCH_LIST` if needed.
