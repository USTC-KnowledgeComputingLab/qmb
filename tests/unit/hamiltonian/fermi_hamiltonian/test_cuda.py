"""CUDA regression tests (requires GPU and compiled .so).

Covers all four operations: diagonal, apply_within (forward+backward),
find_all_relative_configs, find_topk_relative_configs.
All tests compare CUDA output against JAX fallback.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from qmp.hamiltonian.fermi_hamiltonian._hamiltonian import FermiHamiltonian
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_jax import (
    apply_within_subspace as jax_apply_within,
)
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_jax import (
    compute_diagonal_within_subspace as jax_diag,
)
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_jax import (
    find_all_relative_configs as jax_find_all,
)
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_jax import (
    find_topk_relative_configs as jax_find_topk,
)
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_prepare import prepare

cuda_available = any("cuda" in str(d).lower() for d in jax.devices())
pytestmark = pytest.mark.skipif(not cuda_available, reason="No CUDA device available")


def _hopping_hamiltonian() -> FermiHamiltonian:
    return FermiHamiltonian({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4, devices=["localhost:cuda:0"])


def _diagonal_hamiltonian() -> FermiHamiltonian:
    return FermiHamiltonian({((0, 1), (0, 0)): 2.0 + 0j}, n_qubits=8, devices=["localhost:cuda:0"])


def _mixed_hamiltonian() -> FermiHamiltonian:
    return FermiHamiltonian(
        {
            ((1, 1), (0, 0)): -1.0 + 0j,
            ((0, 1), (0, 0)): 2.0 + 0j,
        },
        n_qubits=8,
        devices=["localhost:cuda:0"],
    )


def _configs_4() -> jax.Array:
    return jnp.array([[0b10], [0b01], [0b11], [0b00]], dtype=jnp.uint8)


# ---- diagonal ----


def test_cuda_diagonal_hopping() -> None:
    """CUDA diagonal for hopping-only model vs JAX fallback."""
    h = _hopping_hamiltonian()
    configs = _configs_4()
    cuda_result = h.compute_diagonal_within_subspace(configs)
    jax_masks = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    jax_result = jax_diag(configs, *jax_masks)
    assert jnp.allclose(jax.device_put(cuda_result, jax.devices("cpu")[0]), jax_result, rtol=1e-12)


def test_cuda_diagonal_with_n0() -> None:
    """CUDA diagonal with n_0 term: configs with bit0=1 get 2.0."""
    h = _diagonal_hamiltonian()
    configs = jnp.array([[0b01], [0b00]], dtype=jnp.uint8)
    cuda_result = h.compute_diagonal_within_subspace(configs)
    jax_masks = prepare({((0, 1), (0, 0)): 2.0 + 0j}, n_qubits=8)
    jax_result = jax_diag(configs, *jax_masks)
    assert jnp.allclose(jax.device_put(cuda_result, jax.devices("cpu")[0]), jax_result, rtol=1e-12)


def test_cuda_diagonal_mixed() -> None:
    """CUDA diagonal with hopping + n_0 term."""
    h = _mixed_hamiltonian()
    configs = _configs_4()
    cuda_result = h.compute_diagonal_within_subspace(configs)
    jax_masks = prepare({((1, 1), (0, 0)): -1.0 + 0j, ((0, 1), (0, 0)): 2.0 + 0j}, n_qubits=8)
    jax_result = jax_diag(configs, *jax_masks)
    assert jnp.allclose(jax.device_put(cuda_result, jax.devices("cpu")[0]), jax_result, rtol=1e-12)


# ---- apply_within ----


def test_cuda_apply_within_forward() -> None:
    """CUDA apply_within forward vs JAX fallback."""
    h = _hopping_hamiltonian()
    configs = _configs_4()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    cuda_result = h.apply_within_subspace(configs, psi_i, configs, direction=0)
    jax_masks = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    jax_result = jax_apply_within(configs, psi_i, configs, *jax_masks, direction=0)
    assert jnp.allclose(jax.device_put(cuda_result, jax.devices("cpu")[0]), jax_result, rtol=1e-12)


def test_cuda_apply_within_backward() -> None:
    """CUDA apply_within backward vs JAX fallback."""
    h = _hopping_hamiltonian()
    configs = _configs_4()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    cuda_result = h.apply_within_subspace(configs, psi_i, configs, direction=1)
    jax_masks = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    jax_result = jax_apply_within(configs, psi_i, configs, *jax_masks, direction=1)
    assert jnp.allclose(jax.device_put(cuda_result, jax.devices("cpu")[0]), jax_result, rtol=1e-12)


# ---- find_all ----


def test_cuda_find_all() -> None:
    """CUDA find_all vs JAX fallback."""
    h = _hopping_hamiltonian()
    configs = _configs_4()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    _c_keys, _c_vals, c_cnt = h.find_all_relative_configs(configs, psi_i, exclude, hash_capacity=100)
    jax_masks = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    _j_keys, _j_vals, j_cnt = jax_find_all(configs, psi_i, exclude, *jax_masks, hash_capacity=100)
    assert int(c_cnt) == int(j_cnt)
    # note: CUDA and JAX may produce results in different order due to hash table layout


def test_cuda_hash_table_overflow_retry() -> None:
    """Small capacity forces overflow → should not crash."""
    h = _hopping_hamiltonian()
    configs = jnp.array([[0b10], [0b01]], dtype=jnp.uint8)
    psi_i = jnp.ones((2, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    _new_c, _new_p, cnt = h.find_all_relative_configs(configs, psi_i, exclude, hash_capacity=2)
    assert int(cnt) >= 0


# ---- find_topk ----


def test_cuda_find_topk() -> None:
    """CUDA find_topk vs JAX fallback."""
    h = _hopping_hamiltonian()
    configs = _configs_4()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    cuda_result = h.find_topk_relative_configs(configs, psi_i, 2, exclude)
    jax_masks = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    jax_result = jax_find_topk(configs, psi_i, 2, exclude, *jax_masks)
    assert cuda_result.shape == jax_result.shape
    assert cuda_result.shape == (2, configs.shape[1])
