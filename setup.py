import ast
import re
import os
import subprocess
import setuptools
import importlib

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_dir = os.path.dirname(os.path.realpath(__file__))


# Wheel specific: the wheels only include the SO name of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath("lib")
    for file in path.rglob("libnvshmem_host.so.*"):
        return file.name
    raise ModuleNotFoundError("libnvshmem_host.so not found")


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
    nvshmem_dir = os.getenv("NVSHMEM_DIR", None)
    nvshmem_host_lib = "libnvshmem_host.so"
    if nvshmem_dir is None:
        nvshmem_dir = importlib.util.find_spec(
            "nvidia.nvshmem"
        ).submodule_search_locations[0]
        nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
        import nvidia.nvshmem as nvshmem  # noqa: F401
    assert os.path.exists(
        nvshmem_dir
    ), f"The specified NVSHMEM directory does not exist: {nvshmem_dir}"

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
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = []

    # NVSHMEM flags
    include_dirs.extend([f"{nvshmem_dir}/include"])
    library_dirs.extend([f"{nvshmem_dir}/lib"])
    nvcc_dlink.extend(["-dlink", f"-L{nvshmem_dir}/lib", "-lnvshmem_device"])
    extra_link_args.extend(
        [
            "-lcuda",
            f"-l:{nvshmem_host_lib}",
            "-l:libnvshmem_device.a",
            f"-Wl,-rpath,{nvshmem_dir}/lib",
            "-Wl,--allow-multiple-definition",
        ]
    )

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
                "Only SM90 (Hopper) and SM100 (Blackwell) are supported. "
                "Set TORCH_CUDA_ARCH_LIST manually to override."
            )

    # CUDA 12 flags
    nvcc_flags.extend(
        [
            # "-rdc=true",
            "--ptxas-options=--register-usage-level=10"
        ]
    )

    # Disable aggressive PTX instructions
    if int(os.getenv("DISABLE_AGGRESSIVE_PTX_INSTRS", "1")):
        cxx_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")
        nvcc_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")

    # Put them together
    extra_compile_args = {
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args["nvcc_dlink"] = nvcc_dlink

    # Summary
    print("Build summary:")
    print(f" > Sources: {sources}")
    print(f" > Includes: {include_dirs}")
    print(f" > Libraries: {library_dirs}")
    print(f" > Compilation flags: {extra_compile_args}")
    print(f" > Link flags: {extra_link_args}")
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f" > NVSHMEM path: {nvshmem_dir}")
    print()

    setuptools.setup(
        name="ultra_ep",
        version=get_package_version(),
        packages=setuptools.find_packages(include=["ultra_ep", "ultra_ep.*"]),
        ext_modules=[
            CUDAExtension(
                name="ultra_ep._C",
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                sources=sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        ],
        cmdclass={
            "build_ext": BuildExtension,
        },
    )
