"""CPU-path tests for the FermiHamiltonian Python layer (no GPU required).

test_fallback.py exercises the raw ``_jax_*`` functions directly. This module
instead drives the ``FermiHamiltonian`` class on a CPU device (``_use_cuda`` is
False, so it routes to the JAX fallback), covering the Python-layer logic that
is otherwise only touched on GPU:

- backend selection (cpu device -> fallback, never compiles nvcc);
- diagonal-term pre-filtering (only flip_mask==0 terms are kept/passed);
- operator forwarding + output shapes for all four operations;
- the find_all overflow/return contract and find_topk argsort wrapper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qmp.hamiltonian.fermi_hamiltonian._hamiltonian import FermiHamiltonian
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_jax import (
    compute_diagonal_within_subspace as jax_diag,
)
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_prepare import prepare

_CPU = ["localhost:cpu:0"]


def _c4() -> jax.Array:
    return jnp.array([[0b10], [0b01], [0b11], [0b00]], dtype=jnp.uint8)


# ---- backend selection ----


def test_cpu_device_uses_fallback() -> None:
    """A CPU device selects the pure-JAX fallback, not the CUDA kernel."""
    h = FermiHamiltonian({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4, devices=_CPU)
    assert h._use_cuda is False


# ---- #4 diagonal pre-filter ----


def test_diagonal_prefilter_counts_only_diagonal_terms() -> None:
    """Only flip_mask==0 terms are retained in the diagonal subset."""
    # hopping term (flip != 0) + number term (flip == 0) -> exactly 1 diagonal.
    h = FermiHamiltonian(
        {((1, 1), (0, 0)): -1.0 + 0j, ((0, 1), (0, 0)): 2.0 + 0j},
        n_qubits=4,
        devices=_CPU,
    )
    assert h._diag_count == 1
    assert int(h._diag_coef.shape[0]) == 1


def test_diagonal_prefilter_matches_unfiltered_fallback() -> None:
    """Passing only the diagonal subset gives the same result as the full set.

    Non-diagonal terms contribute zero to the diagonal, so the class-level
    pre-filtered diagonal must equal the raw fallback fed with all terms.
    """
    hamiltonian: dict[tuple[tuple[int, int], ...], complex] = {
        ((1, 1), (0, 0)): -1.0 + 0j,  # hopping, non-diagonal
        ((0, 1), (0, 0)): 2.0 + 0j,  # number, diagonal
        ((1, 1), (1, 0)): 3.0 + 0j,  # number, diagonal
    }
    h = FermiHamiltonian(hamiltonian, n_qubits=4, devices=_CPU)
    configs = _c4()
    filtered = h.compute_diagonal_within_subspace(configs)
    full = jax_diag(configs, *prepare(hamiltonian, n_qubits=4))
    assert jnp.allclose(filtered, full, rtol=1e-12)


def test_diagonal_all_nondiagonal_gives_zero() -> None:
    """A Hamiltonian with only hopping terms has an all-zero diagonal."""
    h = FermiHamiltonian({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4, devices=_CPU)
    assert h._diag_count == 0
    result = h.compute_diagonal_within_subspace(_c4())
    assert jnp.allclose(result, jnp.zeros((4, 2), dtype=jnp.float64))


# ---- operator forwarding / shapes ----


def test_apply_within_forward_backward_shapes() -> None:
    """Forward output lives on configs_j, backward on configs_i."""
    h = FermiHamiltonian({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4, devices=_CPU)
    configs_i = _c4()
    configs_j = jnp.array([[0b10], [0b01]], dtype=jnp.uint8)
    psi_fwd = jnp.ones((4, 2), dtype=jnp.float64)
    psi_bwd = jnp.ones((2, 2), dtype=jnp.float64)
    assert h.apply_within_subspace(configs_i, psi_fwd, configs_j, direction=0).shape == (2, 2)
    assert h.apply_within_subspace(configs_i, psi_bwd, configs_j, direction=1).shape == (4, 2)


def test_find_all_returns_count_and_amplitudes() -> None:
    """find_all returns (configs, psi, count) with a positive count for hopping."""
    h = FermiHamiltonian({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4, devices=_CPU)
    configs = _c4()
    psi = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    new_configs, new_psi, count = h.find_all_relative_configs(configs, psi, exclude, hash_capacity=64)
    assert int(count) > 0
    assert new_configs.shape == (64, 1)
    assert new_psi.shape == (64, 2)


def test_find_topk_returns_k_rows() -> None:
    """find_topk returns exactly count_selected rows of packed configs."""
    h = FermiHamiltonian({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4, devices=_CPU)
    configs = _c4()
    psi = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    selected = h.find_topk_relative_configs(configs, psi, 2, exclude)
    assert selected.shape == (2, 1)
