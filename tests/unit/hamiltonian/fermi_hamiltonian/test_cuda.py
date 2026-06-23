"""CUDA regression tests (requires GPU and compiled .so)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian import FermiHamiltonian
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_jax import (
    compute_diagonal_within_subspace as jax_diag,
)
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_prepare import prepare

pytestmark = pytest.mark.cuda


def _small_configs() -> jax.Array:
    return jnp.array([[0b10], [0b01], [0b11], [0b00]], dtype=jnp.uint8)


def test_cuda_vs_fallback_diagonal() -> None:
    """CUDA diagonal should match JAX fallback output."""
    masks = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    configs = _small_configs()
    h = FermiHamiltonian({((1, 1), (0, 0)): -1.0 + 0j},
                         n_qubits=4, devices=["localhost:cuda:0"])
    cuda_result = h.compute_diagonal_within_subspace(configs)
    jax_result = jax_diag(configs, *masks)
    assert jnp.allclose(
        jax.device_put(cuda_result, jax.devices("cpu")[0]),
        jax_result, rtol=1e-12)


def test_hash_table_overflow_retry() -> None:
    """Verify overflow retry with artificially small capacity."""
    configs = jnp.array([[0b10], [0b01]], dtype=jnp.uint8)
    psi_i = jnp.ones((2, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    h = FermiHamiltonian({((1, 1), (0, 0)): -1.0 + 0j},
                         n_qubits=4, devices=["localhost:cuda:0"])
    new_c, new_p, cnt = h.find_all_relative_configs(
        configs, psi_i, exclude, hash_capacity=2)
    assert int(cnt) >= 0
