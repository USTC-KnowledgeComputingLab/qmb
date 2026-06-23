"""End-to-end tests for pure JAX fallback operations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_jax import (
    apply_within_subspace,
    compute_diagonal_within_subspace,
    find_all_relative_configs,
    find_topk_relative_configs,
)
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_prepare import prepare


def _small_hamiltonian_and_configs() -> tuple[tuple, jax.Array]:
    """4-qubit hopping model + 4 configs."""
    h = {((1, 1), (0, 0)): -1.0 + 0j}
    masks = prepare(h, n_qubits=4)
    configs = jnp.array([
        [0b00000010],
        [0b00000001],
        [0b00000011],
        [0b00000000],
    ], dtype=jnp.uint8)
    return masks, configs


def _h_with_diagonal() -> tuple[tuple, jax.Array]:
    """4-qubit model with diagonal term n_0."""
    h = {
        ((1, 1), (0, 0)): -1.0 + 0j,
        ((0, 1), (0, 0)): 2.0 + 0j,  # n_0 operator (diagonal)
    }
    masks = prepare(h, n_qubits=8)
    configs = jnp.array([
        [0b00000010],
        [0b00000001],
        [0b00000011],
        [0b00000000],
    ], dtype=jnp.uint8)
    return masks, configs


def test_diagonal_exact() -> None:
    """Compute diagonal with terms that include a diagonal n_0."""
    masks, configs = _h_with_diagonal()
    psi = compute_diagonal_within_subspace(configs, *masks)
    # configs with bit0=1 should have diagonal 2.0 from n_0 term
    assert psi.shape == (4, 2)
    # config[0]=0b10 (bit0=0) → 0.0
    # config[1]=0b01 (bit0=1) → 2.0
    assert abs(float(psi[0, 0])) < 1e-10
    assert abs(float(psi[1, 0]) - 2.0) < 1e-10
    assert abs(float(psi[2, 0]) - 2.0) < 1e-10  # 0b11 → bit0=1
    assert abs(float(psi[3, 0])) < 1e-10  # 0b00 → bit0=0


def test_apply_within_forward_backward() -> None:
    """Forward and backward should produce consistent shapes and values."""
    masks, configs = _small_hamiltonian_and_configs()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    forward = apply_within_subspace(configs, psi_i, configs, *masks, direction=0)
    backward = apply_within_subspace(configs, psi_i, configs, *masks, direction=1)
    assert forward.shape == (4, 2)
    assert backward.shape == (4, 2)
    assert jnp.any(jnp.abs(forward) > 0)
    assert jnp.any(jnp.abs(backward) > 0)


def test_find_all_dedup() -> None:
    """find_all_relative_configs should deduplicate and accumulate."""
    masks, configs = _small_hamiltonian_and_configs()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    new_c, new_p, cnt = find_all_relative_configs(
        configs, psi_i, exclude, *masks, hash_capacity=100)
    assert int(cnt) >= 0
    # with a hopping term, should produce some new configs
    assert int(cnt) > 0


def test_find_topk() -> None:
    """find_topk_relative_configs should return K configs."""
    masks, configs = _small_hamiltonian_and_configs()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    result = find_topk_relative_configs(configs, psi_i, 2, exclude, *masks)
    assert result.shape == (2, configs.shape[1])
    assert jnp.any(result != 0)
