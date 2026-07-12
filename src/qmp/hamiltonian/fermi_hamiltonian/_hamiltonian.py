"""Python-layer Fermi Hamiltonian with JAX FFI integration."""

from __future__ import annotations

import logging
import typing

import jax
import jax.ffi
import jax.numpy as jnp

from ._hamiltonian_cuda_loader import load_cuda_module  # optional: needs nvcc to compile .so
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

logger = logging.getLogger(__name__)

# Max capacity-doubling retries when the find_all CUDA hash table overflows.
_MAX_HASH_RETRIES = 8


def _register_ffi(n_qubytes: int) -> None:
    """Compile and register the CUDA FFI targets for the given n_qubytes.

    Raises on failure (missing nvcc, compile error, registration error). The
    caller decides whether to invoke this — it is only called when the target
    device is a CUDA GPU, so any failure here is a real environment error and
    must not be silently downgraded to the (orders-of-magnitude slower) JAX
    fallback.
    """
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
        # 预过滤: 分离对角 term (flip_mask == 0) 用于 compute_diagonal。
        # 只有不改变构型的 term 对对角有贡献; 对角 term 通常仅占 1-10%;
        # 预先切片出子集可让 compute_diagonal 跳过 90-99% 无用 (term, config) 对。
        fm_sum = jnp.sum(self._flip_mask, axis=1)
        diag_idx = jnp.where(fm_sum == 0)[0]
        self._diag_create_mask = self._create_mask[diag_idx]
        self._diag_annihilate_mask = self._annihilate_mask[diag_idx]
        self._diag_flip_mask = self._flip_mask[diag_idx]
        self._diag_parity_mask = self._parity_mask[diag_idx]
        self._diag_parity_const = self._parity_const[diag_idx]
        self._diag_coef = self._coef[diag_idx]
        self._diag_count = int(diag_idx.shape[0])
        # 后端选择由目标设备平台决定: CUDA 设备 → CUDA kernel; 其余 → 纯 JAX。
        # CUDA 设备下若编译/注册失败则抛错; 绝不静默退化到跑不动的 fallback。
        self._use_cuda = self._device.platform == "gpu"
        if self._use_cuda:
            _register_ffi(self._n_qubytes)
        # 哈希表跨调用缓存: apply_within 在 configs_j 不变时复用
        self._apply_hash_cache: tuple[int, typing.Any] | None = None
        logger.info(
            "FermiHamiltonian: %d terms (%d diagonal), %d qubits, cuda=%s",
            int(self._coef.shape[0]),
            self._diag_count,
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
        # 仅传入对角 term 子集 (flip_mask == 0)。kernel/fallback 内部仍会检查
        # flip_mask==0; 但传子集可跳过绝大多数非对角 term 的无用遍历。
        if self._use_cuda:
            return jax.ffi.ffi_call(
                target,
                jax.ShapeDtypeStruct((batch_size, 2), jnp.float64),
                vmap_method="broadcast_all",
            )(
                self._to_dev(configs),
                self._to_dev(self._diag_create_mask),
                self._to_dev(self._diag_annihilate_mask),
                self._to_dev(self._diag_flip_mask),
                self._to_dev(self._diag_parity_mask),
                self._to_dev(self._diag_parity_const),
                self._to_dev(self._diag_coef),
            )
        return _jax_compute_diagonal_within_subspace(
            self._to_dev(configs),
            self._to_dev(self._diag_create_mask),
            self._to_dev(self._diag_annihilate_mask),
            self._to_dev(self._diag_flip_mask),
            self._to_dev(self._diag_parity_mask),
            self._to_dev(self._diag_parity_const),
            self._to_dev(self._diag_coef),
        )

    def apply_within_subspace(
        self,
        configs_i: jax.Array,
        psi_i: jax.Array,
        configs_j: jax.Array,
        *,
        direction: int = 0,
    ) -> jax.Array:
        target = f"qmp_apply_within_subspace_{self._n_qubytes}"
        # Output lives on the destination subspace: forward (dir 0) -> configs_j,
        # backward (dir 1, H^dagger) -> configs_i.
        batch_size_dst = configs_j.shape[0] if direction == 0 else configs_i.shape[0]
        operands = (
            self._to_dev(configs_i),
            self._to_dev(psi_i),
            self._to_dev(configs_j),
            self._to_dev(self._create_mask),
            self._to_dev(self._annihilate_mask),
            self._to_dev(self._flip_mask),
            self._to_dev(self._parity_mask),
            self._to_dev(self._parity_const),
            self._to_dev(self._coef),
        )
        if self._use_cuda:
            return jax.ffi.ffi_call(
                target,
                jax.ShapeDtypeStruct((batch_size_dst, 2), jnp.float64),
                vmap_method="broadcast_all",
            )(*operands, direction=int(direction))
        return _jax_apply_within_subspace(*operands, direction)

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
            # The kernel's open-addressing table can overflow (probe cap hit). It
            # returns an overflow flag; on overflow we double the capacity and
            # retry so no config is silently dropped.
            configs_i_d = self._to_dev(configs_i)
            psi_i_d = self._to_dev(psi_i)
            configs_exclude_d = self._to_dev(configs_exclude)
            create_mask_d = self._to_dev(self._create_mask)
            annihilate_mask_d = self._to_dev(self._annihilate_mask)
            flip_mask_d = self._to_dev(self._flip_mask)
            parity_mask_d = self._to_dev(self._parity_mask)
            parity_const_d = self._to_dev(self._parity_const)
            coef_d = self._to_dev(self._coef)
            capacity = int(hash_capacity)
            for _ in range(_MAX_HASH_RETRIES):
                new_configs, psi_j, count, overflow = jax.ffi.ffi_call(
                    target,
                    (
                        jax.ShapeDtypeStruct((capacity, n_qubytes_dim), jnp.uint8),
                        jax.ShapeDtypeStruct((capacity, 2), jnp.float64),
                        jax.ShapeDtypeStruct((), jnp.int32),
                        jax.ShapeDtypeStruct((), jnp.int32),
                    ),
                    vmap_method="broadcast_all",
                )(
                    configs_i_d,
                    psi_i_d,
                    configs_exclude_d,
                    create_mask_d,
                    annihilate_mask_d,
                    flip_mask_d,
                    parity_mask_d,
                    parity_const_d,
                    coef_d,
                    hash_capacity=capacity,
                )
                if int(overflow) == 0:
                    return new_configs, psi_j, count
                capacity *= 2
            raise RuntimeError(
                f"find_all_relative_configs hash table overflowed after {_MAX_HASH_RETRIES} "
                f"retries (final capacity {capacity})."
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
            # The kernel fills a hash table of capacity 2*count_selected keyed by
            # config, then returns the raw (keys, weights) table. The top-K
            # selection (argsort) happens here to match the JAX fallback exactly.
            capacity = count_selected * 2
            table_keys, table_weights = jax.ffi.ffi_call(
                target,
                (
                    jax.ShapeDtypeStruct((capacity, n_qubytes_dim), jnp.uint8),
                    jax.ShapeDtypeStruct((capacity,), jnp.float64),
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
                count_selected=int(count_selected),
            )
            order = jnp.argsort(table_weights)[::-1][:count_selected]
            return table_keys[order]
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
