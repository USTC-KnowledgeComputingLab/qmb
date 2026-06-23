"""CUDA kernel JIT compilation and caching."""

from __future__ import annotations

import ctypes
import logging
import subprocess
from pathlib import Path

import jaxlib
import platformdirs

logger = logging.getLogger(__name__)

_SOURCE_DIR = Path(__file__).resolve().parent
_THIRD_PARTIES_DIR = Path(__file__).resolve().parents[1] / "third_parties"


def load_cuda_module(n_qubytes: int) -> ctypes.CDLL:
    """Compile and load a CUDA shared library for the given n_qubytes.

    The library is cached in ~/.cache/qmp/kclab/{key}/lib.so.
    On first call for a given n_qubytes, nvcc compiles _hamiltonian_cuda.cu
    with -DN_QUBYTES. Subsequent calls load the cached .so via ctypes.

    Parameters
    ----------
    n_qubytes : int
        ceil(n_qubits/8), passed as -DN_QUBYTES to nvcc.

    Returns
    -------
    ctypes.CDLL
        The loaded shared library.
    """
    key = f"qmp_hamiltonian_{n_qubytes}"
    cache_dir = platformdirs.user_cache_path("qmp", "kclab") / key
    so_path = cache_dir / "lib.so"

    if not so_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        jax_include = str(Path(jaxlib.__file__).parent / "include")

        source = _SOURCE_DIR / "_hamiltonian_cuda.cu"
        include_cuco = _THIRD_PARTIES_DIR / "cuco" / "include"
        cmd = [
            "nvcc",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            f"-I{jax_include}",
            f"-I{include_cuco}",
            f"-DN_QUBYTES={n_qubytes}",
            "-std=c++20",
            "-O3",
            "--use_fast_math",
            "-arch=native",
            "-o",
            str(so_path),
            str(source),
        ]
        logger.info("Compiling CUDA kernel: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        logger.info("CUDA kernel compiled to %s", so_path)

    lib = ctypes.cdll.LoadLibrary(str(so_path))
    return lib
