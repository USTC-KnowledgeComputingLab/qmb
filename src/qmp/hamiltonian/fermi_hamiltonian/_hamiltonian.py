"""Python-layer Fermi Hamiltonian with JAX FFI integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax
import jax.ffi
import jax.numpy as jnp

from ._hamiltonian_jax import (
    apply_within_subspace as _jax_apply_within_subspace,
)
from ._hamiltonian_jax import (
    compute_diagonal_within_subspace as _jax_compute_diagonal_within_subspace,
)
from ._hamiltonian_jax import (
    find_all_relative_configs as _jax_find_all_relative_configs,
)
from ._hamiltonian_jax import (
    find_topk_relative_configs as _jax_find_topk_relative_configs,
)
from ._hamiltonian_prepare import prepare

if TYPE_CHECKING:
    pass


from ._hamiltonian_cuda_loader import load_cuda_module  # optional: needs nvcc to compile .so

logger = logging.getLogger(__name__)

def _try_register_ffi(n_qubytes: int) -> bool:
    """Attempt to load and register CUDA FFI targets for the given n_qubytes."""
    try:
        lib = load_cuda_module(n_qubytes=n_qubytes)
        targets = {
            f"qmp_compute_diagonal_within_subspace_{n_qubytes}": "ComputeDiagonalWithinSubspace",
            f"qmp_apply_within_subspace_{n_qubytes}": "ApplyWithinSubspace",
            f"qmp_find_all_relative_configs_{n_qubytes}": "FindAllRelativeConfigs",
            f"qmp_find_topk_relative_configs_{n_qubytes}": "FindTopKRelativeConfigs",
        }
        for name, sym in targets.items():
            handler = getattr(lib, sym)
            jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(handler), platform="CUDA")
        logger.info("CUDA FFI targets registered for n_qubytes=%d.", n_qubytes)
        return True
    except Exception:
        logger.info("CUDA FFI targets not available; using pure JAX fallback.")
        return False


class FermiHamiltonian:
    """Stores preprocessed bit-mask Hamiltonian and exposes four operations."""

    def _to_dev(self, arr: jax.Array) -> jax.Array:
        return jax.device_put(arr, self._device)

    def __init__(
        self,
        hamiltonian: dict[tuple[tuple[int, int], ...], complex],
        *,
        n_qubits: int,
        devices: list[str],
    ) -> None:
        self._n_qubits = n_qubits
        self._n_qubytes = (n_qubits + 7) // 8
        self._device = self._parse_device(devices)
        arrays = prepare(hamiltonian, n_qubits)
        (
            self._create_mask,
            self._annihilate_mask,
            self._flip_mask,
            self._parity_mask,
            self._parity_const,
            self._coef,
        ) = arrays
        order = jnp.argsort(self._coef[:, 0] ** 2 + self._coef[:, 1] ** 2)[::-1]
        self._create_mask = self._create_mask[order]
        self._annihilate_mask = self._annihilate_mask[order]
        self._flip_mask = self._flip_mask[order]
        self._parity_mask = self._parity_mask[order]
        self._parity_const = self._parity_const[order]
        self._coef = self._coef[order]
        # 预过滤: 分离对角 term (flip_mask == 0) 用于 compute_diagonal
        fm_sum = jnp.sum(self._flip_mask, axis=1)
        self._diag_idx = jnp.where(fm_sum == 0)[0]
        self._use_cuda = _try_register_ffi(self._n_qubytes)
        # 哈希表跨调用缓存: apply_within 在 configs_j 不变时复用
        self._apply_hash_cache: tuple[int, Any] | None = None
        logger.info(
            "FermiHamiltonian: %d terms (%d diagonal), %d qubits, cuda=%s",
            int(self._coef.shape[0]),
            len(self._diag_idx),
            n_qubits,
            self._use_cuda,
        )

    @staticmethod
    def _parse_device(devices: list[str]) -> jax.Device:
        device_str = devices[0]
        parts = device_str.split(":")
        if parts[1] == "cpu":
            return jax.devices("cpu")[0]
        if parts[1] == "cuda":
            idx = int(parts[2]) if len(parts) == 3 else 0
            return jax.devices("cuda")[idx]
        raise ValueError(f"Invalid device: {device_str}")

    def compute_diagonal_within_subspace(self, configs: jax.Array) -> jax.Array:
        batch_size = configs.shape[0]
        target = f"qmp_compute_diagonal_within_subspace_{self._n_qubytes}"
        if self._use_cuda:
            return jax.ffi.ffi_call(
                target,
                jax.ShapeDtypeStruct((batch_size, 2), jnp.float64),
                vmap_method="broadcast_all",
            )(
                self._to_dev(configs),
                self._to_dev(self._create_mask),
                self._to_dev(self._annihilate_mask),
                self._to_dev(self._flip_mask),
                self._to_dev(self._parity_mask),
                self._to_dev(self._parity_const),
                self._to_dev(self._coef),
            )
        return _jax_compute_diagonal_within_subspace(
            self._to_dev(configs),
            self._to_dev(self._create_mask),
            self._to_dev(self._annihilate_mask),
            self._to_dev(self._flip_mask),
            self._to_dev(self._parity_mask),
            self._to_dev(self._parity_const),
            self._to_dev(self._coef),
        )

    def apply_within_subspace(
        self,
        configs_i: jax.Array,
        psi_i: jax.Array,
        configs_j: jax.Array,
        *,
        direction: int = 0,
    ) -> jax.Array:
        batch_size_j = configs_j.shape[0]
        target = f"qmp_apply_within_subspace_{self._n_qubytes}"
        inputs = (
            self._to_dev(configs_i),
            self._to_dev(psi_i),
            self._to_dev(configs_j),
            self._to_dev(self._create_mask),
            self._to_dev(self._annihilate_mask),
            self._to_dev(self._flip_mask),
            self._to_dev(self._parity_mask),
            self._to_dev(self._parity_const),
            self._to_dev(self._coef),
            direction,
        )
        if self._use_cuda:
            return jax.ffi.ffi_call(
                target,
                jax.ShapeDtypeStruct((batch_size_j, 2), jnp.float64),
                vmap_method="broadcast_all",
            )(*inputs)
        return _jax_apply_within_subspace(*inputs)

    def find_all_relative_configs(
        self,
        configs_i: jax.Array,
        psi_i: jax.Array,
        configs_exclude: jax.Array,
        *,
        hash_capacity: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        n_qubytes_dim = configs_i.shape[1]
        target = f"qmp_find_all_relative_configs_{self._n_qubytes}"
        if self._use_cuda:
            return jax.ffi.ffi_call(  # ty: ignore[invalid-return-type] — ffi_call returns Sequence[Array], actual is tuple
                target,
                (
                    jax.ShapeDtypeStruct((hash_capacity, n_qubytes_dim), jnp.uint8),
                    jax.ShapeDtypeStruct((hash_capacity, 2), jnp.float64),
                    jax.ShapeDtypeStruct((), jnp.int32),
                ),
                vmap_method="broadcast_all",
            )(
                self._to_dev(configs_i),
                self._to_dev(psi_i),
                self._to_dev(configs_exclude),
                self._to_dev(self._create_mask),
                self._to_dev(self._annihilate_mask),
                self._to_dev(self._flip_mask),
                self._to_dev(self._parity_mask),
                self._to_dev(self._parity_const),
                self._to_dev(self._coef),
                hash_capacity,
            )
        return _jax_find_all_relative_configs(
            self._to_dev(configs_i),
            self._to_dev(psi_i),
            self._to_dev(configs_exclude),
            self._to_dev(self._create_mask),
            self._to_dev(self._annihilate_mask),
            self._to_dev(self._flip_mask),
            self._to_dev(self._parity_mask),
            self._to_dev(self._parity_const),
            self._to_dev(self._coef),
            hash_capacity,
        )

    def find_topk_relative_configs(
        self,
        configs_i: jax.Array,
        psi_i: jax.Array,
        count_selected: int,
        configs_exclude: jax.Array,
    ) -> jax.Array:
        n_qubytes_dim = configs_i.shape[1]
        target = f"qmp_find_topk_relative_configs_{self._n_qubytes}"
        if self._use_cuda:
            return jax.ffi.ffi_call(
                target,
                jax.ShapeDtypeStruct((count_selected, n_qubytes_dim), jnp.uint8),
                vmap_method="broadcast_all",
            )(
                self._to_dev(configs_i),
                self._to_dev(psi_i),
                count_selected,
                self._to_dev(configs_exclude),
                self._to_dev(self._create_mask),
                self._to_dev(self._annihilate_mask),
                self._to_dev(self._flip_mask),
                self._to_dev(self._parity_mask),
                self._to_dev(self._parity_const),
                self._to_dev(self._coef),
            )
        return _jax_find_topk_relative_configs(
            self._to_dev(configs_i),
            self._to_dev(psi_i),
            count_selected,
            self._to_dev(configs_exclude),
            self._to_dev(self._create_mask),
            self._to_dev(self._annihilate_mask),
            self._to_dev(self._flip_mask),
            self._to_dev(self._parity_mask),
            self._to_dev(self._parity_const),
            self._to_dev(self._coef),
        )
