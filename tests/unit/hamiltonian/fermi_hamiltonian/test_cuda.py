"""CUDA regression tests (requires GPU and compiled .so)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian import FermiHamiltonian
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_jax import (
    compute_diagonal_within_subspace as jax_diag,
)

pytestmark = pytest.mark.cuda


def _small_hamiltonian_and_configs() -> tuple[FermiHamiltonian, jax.Array]:
    h = FermiHamiltonian({((1, 1), (0, 0)): -1.0 + 0j},
                         n_qubits=4, devices=["localhost:cuda:0"])
    configs = jnp.array([[0b10], [0b01], [0b11], [0b00]], dtype=jnp.uint8)
    return h, configs


def test_cuda_vs_fallback_diagonal() -> None:
    """CUDA diagonal should match JAX fallback output."""
    from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_prepare import prepare
    masks, configs = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4), jnp.array(
        [[0b10], [0b01], [0b11], [0b00]], dtype=jnp.uint8)
    h = FermiHamiltonian({((1, 1), (0, 0)): -1.0 + 0j},
                         n_qubits=4, devices=["localhost:cuda:0"])
    cuda_result = h.compute_diagonal_within_subspace(configs)
    jax_result = jax_diag(configs, *masks)
    assert jnp.allclose(
        jax.device_put(cuda_result, jax.devices("cpu")[0]),
        jax_result, rtol=1e-12)
