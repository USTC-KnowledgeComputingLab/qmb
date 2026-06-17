"""
Python-layer Hamiltonian: JAX FFI binding + term preprocessing.

The Hamiltonian class stores preprocessed bit-mask tensors and exposes four
core operations (diagonal_term, apply_within, list_relative, find_relative)
as JAX-compatible callables via :mod:`jax.ffi`.
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.ffi
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from jax import Array

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 加载编译好的 CUDA 共享库
# ---------------------------------------------------------------------------

_LIB_PATH = os.path.join(os.path.dirname(__file__), "libqmp_hamiltonian.so")
_lib: ctypes.CDLL | None = None

try:
    _lib = ctypes.cdll.LoadLibrary(_LIB_PATH)
except OSError:
    logger.warning("CUDA shared library not found at %s; FFI targets will not be registered.", _LIB_PATH)


def _register_target(name: str, symbol_name: str) -> None:
    """注册一个 XLA FFI target。"""
    if _lib is None:
        return
    try:
        handler = getattr(_lib, symbol_name)
    except AttributeError:
        logger.warning("FFI symbol %s not found in shared library.", symbol_name)
        return
    jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(handler), platform="CUDA")


# 注册四个核心操作的 FFI target
_register_target("qmp_diagonal_term", "DiagonalTerm")
_register_target("qmp_apply_within", "ApplyWithin")
_register_target("qmp_list_relative", "ListRelative")
_register_target("qmp_find_relative", "FindRelative")


# ---------------------------------------------------------------------------
# 预处理模块: 将 Hamiltonian dict 转为 bit-mask tensors
# ---------------------------------------------------------------------------

_preparation_module: Any = None


def _get_preparation_module() -> Any:
    global _preparation_module
    if _preparation_module is None:
        import pybind11  # noqa: F401 — ensure pybind11 is available
        from . import _hamiltonian  # type: ignore[attr-defined]

        _preparation_module = _hamiltonian
    return _preparation_module


def prepare_hamiltonian(
    hamiltonian: dict[tuple[tuple[int, int], ...], complex],
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a Hamiltonian dictionary to bit-mask representation.

    The input dict maps operator sequences to complex coefficients.
    Each key is a tuple of (site_index, kind) pairs, where:
      - site_index: int, the qubit index (0-based)
      - kind: 0 for annihilation, 1 for creation, 2 for identity

    Returns six numpy arrays on CPU:
      - create_mask [T, Q]   uint8
      - annihilate_mask [T, Q]   uint8
      - flip_mask [T, Q]     uint8
      - parity_mask [T, Q]   uint8
      - parity_const [T]     uint8
      - coef [T, 2]          float64
    where T = number of non-zero terms, Q = ceil(n_qubits / 8).
    """
    mod = _get_preparation_module()
    return mod.prepare(hamiltonian, n_qubits)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Hamiltonian 类
# ---------------------------------------------------------------------------


class Hamiltonian:
    """
    Stores preprocessed bit-mask Hamiltonian tensors and exposes four
    core operations as JAX-callable methods.

    The term representation uses bit-operation-optimized masks:
    Each term t is described by create_mask[t], annihilate_mask[t],
    flip_mask[t], parity_mask[t], parity_const[t], and coef[t].
    See AGENTS.md for the algorithmic details.

    Parameters
    ----------
    hamiltonian : dict
        Raw Hamiltonian dict mapping operator tuples → complex coefficients.
    n_qubits : int
        Total number of qubits (orbitals × 2 for fermions).
    devices : list[str]
        Device specification, e.g. ``["localhost:cuda:0"]`` or ``["localhost:cpu:0"]``.
        Multiple devices enable shard_map data parallelism.
    """

    def __init__(
        self,
        hamiltonian: dict[tuple[tuple[int, int], ...], complex],
        *,
        n_qubits: int,
        devices: list[str],
    ) -> None:
        self._device = self._parse_device(devices)
        self._n_qubits = n_qubits
        self._n_qubytes = (n_qubits + 7) // 8

        logger.info("Preprocessing Hamiltonian with %d terms on %d qubits...", len(hamiltonian), n_qubits)
        arrays = prepare_hamiltonian(hamiltonian, n_qubits)
        self._create_mask, self._annihilate_mask, self._flip_mask, self._parity_mask, self._parity_const, self._coef = arrays

        # 按 |coef| 降序排序 (改善收敛)
        order = np.argsort(self._coef[:, 0] ** 2 + self._coef[:, 1] ** 2)[::-1]
        self._create_mask = self._create_mask[order]
        self._annihilate_mask = self._annihilate_mask[order]
        self._flip_mask = self._flip_mask[order]
        self._parity_mask = self._parity_mask[order]
        self._parity_const = self._parity_const[order]
        self._coef = self._coef[order]

        logger.info("Hamiltonian ready: %d terms, %d qubits, device=%s",
                     len(self._coef), n_qubits, self._device)

    @staticmethod
    def _parse_device(devices: list[str]) -> jax.Device:
        if len(devices) > 1:
            raise NotImplementedError("Multiple devices are not yet supported in the base Hamiltonian class. Use shard_map externally.")
        device_str = devices[0]
        parts = device_str.split(":")
        if parts[1] == "cpu":
            return jax.devices("cpu")[0]
        if parts[1] == "cuda":
            device_idx = int(parts[2]) if len(parts) == 3 else 0
            return jax.devices("cuda")[device_idx]
        raise ValueError(f"Invalid device string: {device_str}")

    # ---- JAX-callable operations ----

    def diagonal_term(self, configs: Array) -> Array:
        """
        Compute diagonal Hamiltonian elements.

        Parameters
        ----------
        configs : Array
            ``[B, Q]`` uint8 bit-packed configurations.

        Returns
        -------
        Array
            ``[B, 2]`` float64 (real, imag) diagonal energies.
        """
        B = configs.shape[0]
        return jax.ffi.ffi_call(
            "qmp_diagonal_term",
            jax.ShapeDtypeStruct((B, 2), jnp.float64),
            vmap_method="broadcast_all",
        )(
            _to_device(configs, self._device),
            _to_device(self._create_mask, self._device),
            _to_device(self._annihilate_mask, self._device),
            _to_device(self._flip_mask, self._device),
            _to_device(self._parity_mask, self._device),
            _to_device(self._parity_const, self._device),
            _to_device(self._coef, self._device),
        )

    def apply_within(
        self,
        configs_i: Array,
        psi_i: Array,
        configs_j: Array,
        *,
        direction: int = 0,
    ) -> Array:
        """
        Apply Hamiltonian to psi_i, projecting onto configs_j.

        Parameters
        ----------
        configs_i : Array
            ``[B_i, Q]`` uint8 source configurations.
        psi_i : Array
            ``[B_i, 2]`` float64 (real, imag) amplitudes on configs_i.
        configs_j : Array
            ``[B_j, Q]`` uint8 target configurations.
        direction : int
            ``0`` = forward (H), ``1`` = backward (H^dag).

        Returns
        -------
        Array
            ``[B_j, 2]`` float64 projected amplitudes.
        """
        return jax.ffi.ffi_call(
            "qmp_apply_within",
            jax.ShapeDtypeStruct((configs_j.shape[0], 2), jnp.float64),
            vmap_method="broadcast_all",
        )(
            _to_device(configs_i, self._device),
            _to_device(psi_i, self._device),
            _to_device(configs_j, self._device),
            _to_device(self._create_mask, self._device),
            _to_device(self._annihilate_mask, self._device),
            _to_device(self._flip_mask, self._device),
            _to_device(self._parity_mask, self._device),
            _to_device(self._parity_const, self._device),
            _to_device(self._coef, self._device),
            direction=np.int32(direction),
        )

    def list_relative(
        self,
        configs_i: Array,
        psi_i: Array,
        configs_exclude: Array,
        *,
        hash_capacity: int,
    ) -> tuple[Array, Array, Array]:
        """
        List all unique new configurations reachable via H.

        Parameters
        ----------
        configs_i : Array
            ``[B_i, Q]`` uint8 source configurations.
        psi_i : Array
            ``[B_i, 2]`` float64 amplitudes.
        configs_exclude : Array
            ``[E, Q]`` uint8 configurations to exclude.
        hash_capacity : int
            Pre-allocated hash table capacity (slots).

        Returns
        -------
        tuple[Array, Array, Array]
            ``(new_configs, psi_j, count)`` where new_configs and psi_j are
            pre-allocated to hash_capacity, and count is the actual number.
        """
        Q = configs_i.shape[1]
        new_c, new_p, cnt = jax.ffi.ffi_call(
            "qmp_list_relative",
            (
                jax.ShapeDtypeStruct((hash_capacity, Q), jnp.uint8),
                jax.ShapeDtypeStruct((hash_capacity, 2), jnp.float64),
                jax.ShapeDtypeStruct((), jnp.int32),
            ),
            vmap_method="broadcast_all",
        )(
            _to_device(configs_i, self._device),
            _to_device(psi_i, self._device),
            _to_device(configs_exclude, self._device),
            _to_device(self._create_mask, self._device),
            _to_device(self._annihilate_mask, self._device),
            _to_device(self._flip_mask, self._device),
            _to_device(self._parity_mask, self._device),
            _to_device(self._parity_const, self._device),
            _to_device(self._coef, self._device),
            hash_capacity=np.int32(hash_capacity),
        )
        return new_c, new_p, cnt

    def find_relative(
        self,
        configs_i: Array,
        psi_i: Array,
        count_selected: int,
        configs_exclude: Array,
    ) -> Array:
        """
        Find top-K most important new configurations.

        Parameters
        ----------
        configs_i : Array
            ``[B_i, Q]`` uint8 source configurations.
        psi_i : Array
            ``[B_i, 2]`` float64 amplitudes.
        count_selected : int
            Number of new configurations to select (K).
        configs_exclude : Array
            ``[E, Q]`` uint8 configurations to exclude.

        Returns
        -------
        Array
            ``[K, Q]`` uint8 top-K new configurations.
        """
        Q = configs_i.shape[1]
        return jax.ffi.ffi_call(
            "qmp_find_relative",
            jax.ShapeDtypeStruct((count_selected, Q), jnp.uint8),
            vmap_method="broadcast_all",
        )(
            _to_device(configs_i, self._device),
            _to_device(psi_i, self._device),
            count_selected=np.int32(count_selected),
            _to_device(configs_exclude, self._device),
            _to_device(self._create_mask, self._device),
            _to_device(self._annihilate_mask, self._device),
            _to_device(self._flip_mask, self._device),
            _to_device(self._parity_mask, self._device),
            _to_device(self._parity_const, self._device),
            _to_device(self._coef, self._device),
        )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _to_device(arr: np.ndarray | Array, device: jax.Device) -> Array:
    """Ensure an array is on the target JAX device."""
    if isinstance(arr, np.ndarray):
        return jax.device_put(jnp.asarray(arr), device)
    return jax.device_put(arr, device)
