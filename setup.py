import ast
import re
import os
import subprocess
import setuptools

from pathlib import Path
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_dir = os.path.dirname(os.path.realpath(__file__))


def get_package_version():
    with open(Path(current_dir) / "ultra_ep" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))

    # noinspection PyBroadException
    try:
        status_cmd = ["git", "status", "--porcelain"]
        status_output = subprocess.check_output(status_cmd).decode("ascii").strip()
        if status_output:
            print(
                f"Warning: Git working directory is not clean. Uncommitted changes:\n{status_output}"
            )

        cmd = ["git", "rev-parse", "--short", "HEAD"]
        revision = "+" + subprocess.check_output(cmd).decode("ascii").rstrip()
    except:
        revision = "+local"
    return f"{public_version}{revision}"


def find_cpp_cuda_sources(root_dir="csrc"):
    valid_exts = {".cpp", ".cc", ".cu"}
    source_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if Path(fname).suffix in valid_exts:
                source_files.append(str(Path(dirpath) / fname))
    return source_files


if __name__ == "__main__":
    cxx_flags = [
        "-O3",
        "-Wno-deprecated-declarations",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-reorder",
        "-Wno-attributes",
    ]
    nvcc_flags = ["-O3", "-Xcompiler", "-O3"]

    sources = find_cpp_cuda_sources("csrc")
    include_dirs = ["csrc/"]

    # Auto-detect CUDA arch if not explicitly set
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        compute_cap = result.stdout.strip().splitlines()[0].strip()
        sm = int(compute_cap.replace(".", ""))
        if sm == 90:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
        elif sm == 100:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
        else:
            raise RuntimeError(
                f"Unsupported CUDA compute capability: {compute_cap} (SM{sm}). "
                "Only SM90 and SM100 are supported. "
                "Set TORCH_CUDA_ARCH_LIST manually to override."
            )

    nvcc_flags.extend(["--ptxas-options=--register-usage-level=10"])
    # Compilation workaround for CUDA 13.x
    if torch.version.cuda and torch.version.cuda.startswith("13."):
        include_dirs.extend(["/usr/local/cuda/include/cccl/"])

    # Disable aggressive PTX instructions
    if int(os.getenv("DISABLE_AGGRESSIVE_PTX_INSTRS", "1")):
        cxx_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")
        nvcc_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")

    # Put them together
    extra_compile_args = {
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    }
    # Summary
    print("Build summary:")
    print(f" > Sources: {sources}")
    print(f" > Includes: {include_dirs}")
    print(f" > Compilation flags: {extra_compile_args}")
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print()

    setuptools.setup(
        name="ultra_ep",
        version=get_package_version(),
        packages=setuptools.find_packages(include=["ultra_ep", "ultra_ep.*"]),
        install_requires=["torch>=2.11", "nvidia-ml-py>=12.570"],
        ext_modules=[
            CUDAExtension(
                name="ultra_ep._C",
                include_dirs=include_dirs,
                sources=sources,
                extra_compile_args=extra_compile_args,
            )
        ],
        cmdclass={
            "build_ext": BuildExtension,
        },
    )
