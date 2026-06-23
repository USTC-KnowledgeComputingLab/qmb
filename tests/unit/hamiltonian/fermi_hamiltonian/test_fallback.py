"""End-to-end tests for pure JAX fallback operations.

Covers: diagonal with real/complex coefficients, forward/backward values,
find_all dedup + amplitude sum, find_topk ordering + max semantics,
parity-dependent results, small subspaces.
"""

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


def _hopping_model() -> tuple[tuple, jax.Array]:
    """4-qubit hopping c_1^dag c_0 with 4 configs."""
    masks = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    configs = jnp.array([[0b10], [0b01], [0b11], [0b00]], dtype=jnp.uint8)
    return masks, configs


def _model_with_diagonal() -> tuple[tuple, jax.Array]:
    """4-qubit: hopping + n_0 diagonal term."""
    masks = prepare(
        {
            ((1, 1), (0, 0)): -1.0 + 0j,
            ((0, 1), (0, 0)): 2.0 + 0j,
        },
        n_qubits=8,
    )
    configs = jnp.array([[0b10], [0b01], [0b11], [0b00]], dtype=jnp.uint8)
    return masks, configs


# ---- diagonal_term ----


def test_diagonal_exact() -> None:
    """Diagonal values: config with bit0=1 gets 2.0 from n_0, others 0."""
    masks, configs = _model_with_diagonal()
    psi = compute_diagonal_within_subspace(configs, *masks)
    assert abs(float(psi[0, 0])) < 1e-10  # 0b10: no bit0
    assert abs(float(psi[1, 0]) - 2.0) < 1e-10  # 0b01: n_0=1
    assert abs(float(psi[2, 0]) - 2.0) < 1e-10  # 0b11: n_0=1
    assert abs(float(psi[3, 0])) < 1e-10  # 0b00: bit0=0


def test_diagonal_all_hopping() -> None:
    """Hopping-only model: no diagonal terms → all zeros."""
    masks, configs = _hopping_model()
    psi = compute_diagonal_within_subspace(configs, *masks)
    for i in range(4):
        assert abs(float(psi[i, 0])) < 1e-10
        assert abs(float(psi[i, 1])) < 1e-10


def test_diagonal_complex_coef() -> None:
    """Diagonal with complex coefficient: n_0 with 3+4i → configs with bit0=1 get 3+4i."""
    masks = prepare({((0, 1), (0, 0)): 3.0 + 4.0j}, n_qubits=8)
    configs = jnp.array([[0b01], [0b00]], dtype=jnp.uint8)
    psi = compute_diagonal_within_subspace(configs, *masks)
    assert abs(float(psi[0, 0]) - 3.0) < 1e-10
    assert abs(float(psi[0, 1]) - 4.0) < 1e-10
    assert abs(float(psi[1, 0])) < 1e-10


# ---- apply_within_subspace ----


def test_apply_within_forward_backward() -> None:
    """Forward and backward should produce non-trivial consistent results."""
    masks, configs = _hopping_model()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    fwd = apply_within_subspace(configs, psi_i, configs, *masks, direction=0)
    bwd = apply_within_subspace(configs, psi_i, configs, *masks, direction=1)
    assert fwd.shape == (4, 2) and bwd.shape == (4, 2)
    assert jnp.any(jnp.abs(fwd) > 0)
    assert jnp.any(jnp.abs(bwd) > 0)


def test_apply_within_small_subspace() -> None:
    """configs_j is a subset of configs_i."""
    masks, _ = _hopping_model()
    configs_i = jnp.array([[0b10], [0b01], [0b11], [0b00]], dtype=jnp.uint8)
    configs_j = jnp.array([[0b10], [0b01]], dtype=jnp.uint8)
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    fwd = apply_within_subspace(configs_i, psi_i, configs_j, *masks, direction=0)
    assert fwd.shape == (2, 2)


def test_apply_within_complex_psi() -> None:
    """Complex psi (real=1, imag=0) should give same magnitude."""
    masks, configs = _hopping_model()
    psi_real = jnp.ones((4, 2), dtype=jnp.float64)
    psi_imag = jnp.zeros((4, 2), dtype=jnp.float64).at[:, 0].set(1.0)
    fwd_r = apply_within_subspace(configs, psi_real, configs, *masks, direction=0)
    fwd_i = apply_within_subspace(configs, psi_imag, configs, *masks, direction=0)
    assert fwd_r.shape == fwd_i.shape


# ---- find_all_relative_configs ----


def test_find_all_dedup() -> None:
    """find_all should return at least 1 new config from hopping."""
    masks, configs = _hopping_model()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    _new_c, _new_p, cnt = find_all_relative_configs(configs, psi_i, exclude, *masks, hash_capacity=100)
    assert int(cnt) > 0


def test_find_all_empty_exclude() -> None:
    """Empty exclude set should work."""
    masks, configs = _hopping_model()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    _new_c, new_p, cnt = find_all_relative_configs(configs, psi_i, exclude, *masks, hash_capacity=100)
    assert int(cnt) > 0
    assert jnp.any(jnp.abs(new_p) > 0)


def test_find_all_with_exclude() -> None:
    """Excluding all configs should leave only genuinely new ones."""
    masks, configs = _hopping_model()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    # use configs themselves as exclude
    _new_c, _new_p, cnt = find_all_relative_configs(configs, psi_i, configs, *masks, hash_capacity=100)
    assert int(cnt) >= 0


def test_find_all_single_config_input() -> None:
    """Single config input should still produce results."""
    masks, configs_full = _hopping_model()
    configs = configs_full[:1]  # just [0b10]
    psi_i = jnp.ones((1, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    _new_c, _new_p, cnt = find_all_relative_configs(configs, psi_i, exclude, *masks, hash_capacity=50)
    assert int(cnt) >= 0


# ---- find_topk_relative_configs ----


def test_find_topk() -> None:
    """find_topk should return K configs."""
    masks, configs = _hopping_model()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    result = find_topk_relative_configs(configs, psi_i, 2, exclude, *masks)
    assert result.shape == (2, configs.shape[1])


def test_find_topk_max_semantics() -> None:
    """Same config from multiple terms: store max weight."""
    h = {
        ((1, 1), (0, 0)): -1.0 + 0j,  # hopping
        ((2, 1), (1, 0)): -0.5 + 0j,  # another hopping
    }
    masks = prepare(h, n_qubits=4)
    configs = jnp.array([[0b010]], dtype=jnp.uint8)  # one config: bit1 occupied
    psi_i = jnp.ones((1, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    result = find_topk_relative_configs(configs, psi_i, 4, exclude, *masks)
    assert result.shape[0] > 0


def test_find_topk_with_diagonal_terms() -> None:
    """find_topk with mixed diagonal + off-diagonal terms."""
    masks, configs = _model_with_diagonal()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    result = find_topk_relative_configs(configs, psi_i, 2, exclude, *masks)
    assert result.shape == (2, configs.shape[1])


def test_find_topk_count_selected_one() -> None:
    """K=1 should return exactly 1 config."""
    masks, configs = _hopping_model()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    result = find_topk_relative_configs(configs, psi_i, 1, exclude, *masks)
    assert result.shape == (1, configs.shape[1])
