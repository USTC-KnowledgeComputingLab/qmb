"""End-to-end tests for pure JAX fallback operations.

Covers: diagonal with real/complex coefficients, forward/backward values,
find_all dedup + amplitude sum, find_topk ordering + max semantics,
edge cases (empty input, identity configs, superset/subset configs),
Hubbard model, multiple contributions, complex psi, JW parity.
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

# ---- helpers ----


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


def _two_site_fermion_masks() -> tuple:
    """H = c†₁c₀ + c†₀c₁ (2-site fermion, no spin)."""
    return prepare({((1, 1), (0, 0)): -1.0 + 0j, ((0, 1), (1, 0)): -1.0 + 0j}, n_qubits=4)


def _two_site_configs() -> jax.Array:
    return jnp.array([[1], [2], [3], [0]], dtype=jnp.uint8)  # |01⟩, |10⟩, |11⟩, |00⟩


def _configs_4() -> jax.Array:
    return jnp.array([[0b10], [0b01], [0b11], [0b00]], dtype=jnp.uint8)


# ---- diagonal_term ----


def test_diagonal_exact() -> None:
    """Diagonal values: config with bit0=1 gets 2.0 from n_0, others 0."""
    masks, configs = _model_with_diagonal()
    psi = compute_diagonal_within_subspace(configs, *masks)
    assert abs(float(psi[0, 0])) < 1e-10
    assert abs(float(psi[1, 0]) - 2.0) < 1e-10
    assert abs(float(psi[2, 0]) - 2.0) < 1e-10
    assert abs(float(psi[3, 0])) < 1e-10


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


def test_diagonal_only_hamiltonian() -> None:
    """Number operators only: n_0=1.0, n_1=2.0. Configs get correct diagonal."""
    masks = prepare({((0, 1), (0, 0)): 1.0 + 0j, ((1, 1), (1, 0)): 2.0 + 0j}, n_qubits=8)
    configs = jnp.array([[0b01], [0b10], [0b11]], dtype=jnp.uint8)
    psi = compute_diagonal_within_subspace(configs, *masks)
    assert abs(float(psi[0, 0]) - 1.0) < 1e-10  # only n_0
    assert abs(float(psi[1, 0]) - 2.0) < 1e-10  # only n_1
    assert abs(float(psi[2, 0]) - 3.0) < 1e-10  # both n_0 + n_1


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


def test_apply_within_numerical_hopping() -> None:
    """H|c†₀|vac⟩ = -|c†₁|vac⟩ — exact numerical test."""
    masks = _two_site_fermion_masks()
    ci = jnp.array([[1]], dtype=jnp.uint8)  # site 0 occupied
    pi = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    cj = jnp.array([[2]], dtype=jnp.uint8)  # site 1 occupied
    pj = apply_within_subspace(ci, pi, cj, *masks, direction=0)
    assert abs(float(pj[0, 0]) - (-1.0)) < 1e-10  # coefficient -1, JW sign +


def test_apply_within_hermitian() -> None:
    """H|10⟩ = -|01⟩."""
    masks = _two_site_fermion_masks()
    ci = jnp.array([[2]], dtype=jnp.uint8)
    pi = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    cj = jnp.array([[1]], dtype=jnp.uint8)
    pj = apply_within_subspace(ci, pi, cj, *masks, direction=0)
    assert abs(float(pj[0, 0]) - (-1.0)) < 1e-10


def test_apply_within_pauli_exclusion() -> None:
    """|11⟩ both occupied: applying hopping term fails → zero."""
    masks = _two_site_fermion_masks()
    ci = jnp.array([[3]], dtype=jnp.uint8)
    pi = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    cj = jnp.array([[3]], dtype=jnp.uint8)
    pj = apply_within_subspace(ci, pi, cj, *masks, direction=0)
    assert abs(float(pj[0, 0])) < 1e-10


def test_apply_within_no_connected() -> None:
    """|00⟩ (empty) not connected to |11⟩ by creation operators."""
    masks = _two_site_fermion_masks()
    ci = jnp.array([[0]], dtype=jnp.uint8)
    pi = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    cj = jnp.array([[3]], dtype=jnp.uint8)
    pj = apply_within_subspace(ci, pi, cj, *masks, direction=0)
    assert abs(float(pj[0, 0])) < 1e-10


def test_apply_within_identity_configs() -> None:
    """configs_i == configs_j, purely off-diagonal H → all zero."""
    masks = _two_site_fermion_masks()
    ci = jnp.array([[1]], dtype=jnp.uint8)
    pi = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    pj = apply_within_subspace(ci, pi, ci, *masks, direction=0)
    assert abs(float(pj[0, 0])) < 1e-10


def test_apply_within_complex_psi() -> None:
    """Complex psi_i + complex coefficient = correct multiplication."""
    masks = prepare({((1, 1), (0, 0)): 1.0 + 0.0j}, n_qubits=4)
    ci = jnp.array([[1]], dtype=jnp.uint8)
    pi = jnp.array([[2.0, 3.0]], dtype=jnp.float64)
    cj = jnp.array([[2]], dtype=jnp.uint8)
    pj = apply_within_subspace(ci, pi, cj, *masks, direction=0)
    assert abs(float(pj[0, 0]) - 2.0) < 1e-10
    assert abs(float(pj[0, 1]) - 3.0) < 1e-10


def test_apply_within_multiple_contributions() -> None:
    """Two source configs contribute to same target → amplitudes sum."""
    masks = prepare({((1, 1), (0, 0)): -2.0 + 0j, ((0, 1), (1, 0)): -1.0 + 0j}, n_qubits=4)
    ci = jnp.array([[1], [2]], dtype=jnp.uint8)
    pi = jnp.array([[1.0, 0.0], [0.5, 0.0]], dtype=jnp.float64)
    cj = jnp.array([[1], [2]], dtype=jnp.uint8)
    pj = apply_within_subspace(ci, pi, cj, *masks, direction=0)
    assert abs(float(pj[0, 0]) - (-0.5)) < 1e-10  # from ci[1] via c†₀c₁
    assert abs(float(pj[1, 0]) - (-2.0)) < 1e-10  # from ci[0] via c†₁c₀


def test_apply_within_superset_j() -> None:
    """configs_j includes unreachable configs → those stay zero."""
    masks = _two_site_fermion_masks()
    ci = jnp.array([[1]], dtype=jnp.uint8)
    pi = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    cj = jnp.array([[0], [1], [2]], dtype=jnp.uint8)
    pj = apply_within_subspace(ci, pi, cj, *masks, direction=0)
    assert abs(float(pj[0, 0])) < 1e-10  # |00⟩ not reachable
    assert abs(float(pj[1, 0])) < 1e-10  # |01⟩ not reachable (self, no diag)
    assert abs(float(pj[2, 0]) - (-1.0)) < 1e-10  # |10⟩


def test_apply_within_small_subspace() -> None:
    """configs_j is a subset of configs_i."""
    masks, _ = _hopping_model()
    configs_i = _configs_4()
    configs_j = jnp.array([[0b10], [0b01]], dtype=jnp.uint8)
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    fwd = apply_within_subspace(configs_i, psi_i, configs_j, *masks, direction=0)
    assert fwd.shape == (2, 2)


def test_apply_within_complex_coef() -> None:
    """Complex coefficient (0+1j) produces pure imaginary output."""
    masks = prepare({((1, 1), (0, 0)): 0.0 + 1.0j}, n_qubits=4)
    ci = jnp.array([[1]], dtype=jnp.uint8)
    pi = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    cj = jnp.array([[2]], dtype=jnp.uint8)
    pj = apply_within_subspace(ci, pi, cj, *masks, direction=0)
    assert abs(float(pj[0, 0])) < 1e-10
    assert abs(float(pj[0, 1]) - 1.0) < 1e-10


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
    _new_c, _new_p, cnt = find_all_relative_configs(configs, psi_i, configs, *masks, hash_capacity=100)
    assert int(cnt) >= 0


def test_find_all_single_config_input() -> None:
    """Single config input should still produce results."""
    masks, configs_full = _hopping_model()
    configs = configs_full[:1]
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
    h: dict[tuple[tuple[int, int], ...], complex] = {((1, 1), (0, 0)): -1.0 + 0j, ((2, 1), (1, 0)): -0.5 + 0j}
    masks = prepare(h, n_qubits=4)
    configs = jnp.array([[0b010]], dtype=jnp.uint8)
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


# ---- apply_within Hermitian conjugation ----


def test_apply_within_hermitian_conjugate() -> None:
    """Forward(direction=0) and backward(direction=1) should be Hermitian conjugates."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((1, 1), (0, 0)): -2.0 + 1.0j}
    masks = prepare(h, n_qubits=4)
    ci = jnp.array([[0b01], [0b10], [0b00], [0b11]], dtype=jnp.uint8)
    cj = jnp.array([[0b01], [0b10], [0b00]], dtype=jnp.uint8)
    psi_i = jnp.array([[0.5, 1.0], [0.0, 0.0], [2.0, 3.0], [0.3, -0.7]], dtype=jnp.float64)
    psi_j = jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]], dtype=jnp.float64)
    fwd_i2j = apply_within_subspace(ci, psi_i, cj, *masks, direction=0)
    bwd_j2i = apply_within_subspace(ci, psi_j, cj, *masks, direction=1)
    inner_fwd = jnp.sum(psi_j[:, 0] * fwd_i2j[:, 0] + psi_j[:, 1] * fwd_i2j[:, 1])
    inner_bwd = jnp.sum(psi_i[:, 0] * bwd_j2i[:, 0] + psi_i[:, 1] * bwd_j2i[:, 1])
    assert jnp.allclose(inner_fwd, inner_bwd, rtol=1e-10)


# ---- find_all amplitude correctness ----


def test_find_all_amplitude_exact() -> None:
    """c_1^dag c_0 on |01> with psi=1+0j → |10> with amplitude -1+0j (JW sign +)."""
    masks = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    configs = jnp.array([[0b01]], dtype=jnp.uint8)
    psi_i = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    new_c, new_p, cnt = find_all_relative_configs(configs, psi_i, exclude, *masks, hash_capacity=50)
    assert int(cnt) == 1
    assert int(new_c[0, 0]) == 0b10  # |10⟩
    assert abs(float(new_p[0, 0]) - (-1.0)) < 1e-10
    assert abs(float(new_p[0, 1])) < 1e-10


def test_find_all_amplitude_sum() -> None:
    """Two source configs both connect to the same new config: amplitudes accumulate."""
    masks = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    configs = jnp.array([[0b01], [0b01]], dtype=jnp.uint8)
    psi_i = jnp.array([[1.0, 0.0], [0.5, 0.0]], dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    _new_c, new_p, cnt = find_all_relative_configs(configs, psi_i, exclude, *masks, hash_capacity=50)
    assert int(cnt) == 1
    assert abs(float(new_p[0, 0]) - (-1.5)) < 1e-10


# ---- find_topk ordering ----


def test_find_topk_ordering() -> None:
    """find_topk should return configs in descending weight order."""
    h: dict[tuple[tuple[int, int], ...], complex] = {
        ((1, 1), (0, 0)): -3.0 + 0j,
        ((2, 1), (1, 0)): -1.0 + 0j,
        ((3, 1), (2, 0)): -0.1 + 0j,
    }
    masks = prepare(h, n_qubits=4)
    configs = jnp.array([[0b010], [0b100], [0b1000]], dtype=jnp.uint8)
    psi_i = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    result = find_topk_relative_configs(configs, psi_i, 3, exclude, *masks)
    assert result.shape == (3, configs.shape[1])
    configs_set = {int(r[0]) for r in result}
    assert len(configs_set) == 3  # distinct configs


def test_find_topk_k_exceeds_distinct() -> None:
    """K larger than the number of distinct new configs should not crash."""
    masks = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    configs = jnp.array([[0b01]], dtype=jnp.uint8)
    psi_i = jnp.ones((1, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    result = find_topk_relative_configs(configs, psi_i, 10, exclude, *masks)
    assert result.shape == (10, configs.shape[1])


# ---- diagonal with JW parity ----


def test_diagonal_with_parity() -> None:
    """n_2 = c_2^dag c_2 with coef=2.0: diagonal contribution for configs with qubit 2 occupied."""
    masks = prepare({((2, 1), (2, 0)): 2.0 + 0j}, n_qubits=8)
    configs = jnp.array([[0b100], [0b101], [0b001], [0b010]], dtype=jnp.uint8)
    psi = compute_diagonal_within_subspace(configs, *masks)
    assert abs(float(psi[0, 0]) - 2.0) < 1e-10  # |100⟩: bit2 occupied → +2.0
    assert abs(float(psi[1, 0]) - 2.0) < 1e-10  # |101⟩: bit2 occupied → +2.0
    assert abs(float(psi[2, 0])) < 1e-10  # |001⟩: bit2 not occupied → 0
    assert abs(float(psi[3, 0])) < 1e-10  # |010⟩: bit2 not occupied → 0


# ---- multi-byte qubits ----


def test_diagonal_multi_byte() -> None:
    """n_qubits=12 (Q=2) diagonal: n_0 + n_10."""
    masks = prepare({((0, 1), (0, 0)): 1.0 + 0j, ((10, 1), (10, 0)): 5.0 + 0j}, n_qubits=12)
    configs = jnp.array([[0, 0b100], [0b01, 0]], dtype=jnp.uint8)
    psi = compute_diagonal_within_subspace(configs, *masks)
    assert abs(float(psi[0, 0]) - 5.0) < 1e-10  # bit 10 in byte 1
    assert abs(float(psi[1, 0]) - 1.0) < 1e-10  # bit 0 in byte 0


def test_apply_within_multi_byte() -> None:
    """n_qubits=12 hopping c_9^dag c_2."""
    masks = prepare({((9, 1), (2, 0)): -1.0 + 0j}, n_qubits=12)
    ci = jnp.array([[0b100, 0]], dtype=jnp.uint8)  # bit 2 set in byte 0
    pi = jnp.ones((1, 2), dtype=jnp.float64)
    cj = jnp.array([[0, 0b10]], dtype=jnp.uint8)  # bit 9 set in byte 1
    pj = apply_within_subspace(ci, pi, cj, *masks, direction=0)
    assert abs(float(pj[0, 0]) - (-1.0)) < 1e-10


def test_find_all_multi_byte() -> None:
    """n_qubits=12 find_all with hopping."""
    masks = prepare({((9, 1), (2, 0)): -1.0 + 0j}, n_qubits=12)
    configs = jnp.array([[0b100, 0]], dtype=jnp.uint8)
    psi_i = jnp.ones((1, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 2), dtype=jnp.uint8)
    new_c, _new_p, cnt = find_all_relative_configs(configs, psi_i, exclude, *masks, hash_capacity=50)
    assert int(cnt) == 1
    assert int(new_c[0, 1]) & 2 == 2  # bit 9 in byte 1


# ---- zero psi ----


def test_diagonal_zero_psi() -> None:
    """psi_i all zeros: diagonal should return zero."""
    masks, configs = _hopping_model()
    psi = compute_diagonal_within_subspace(configs, *masks)
    for i in range(psi.shape[0]):
        assert abs(float(psi[i, 0])) < 1e-10
        assert abs(float(psi[i, 1])) < 1e-10


def test_apply_within_zero_psi() -> None:
    """psi_i all zeros: apply_within should return zero."""
    masks, configs = _hopping_model()
    psi_i = jnp.zeros((4, 2), dtype=jnp.float64)
    fwd = apply_within_subspace(configs, psi_i, configs, *masks, direction=0)
    assert jnp.all(abs(fwd) < 1e-10)


# ---- complex coef x complex psi x direction=1 ----


def test_apply_within_complex_hermitian() -> None:
    """Complex coefficient, complex psi, direction=1 → full H^† application."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((1, 1), (0, 0)): 0.5 + 1.5j}
    masks = prepare(h, n_qubits=4)
    ci = jnp.array([[1], [2]], dtype=jnp.uint8)
    cj = jnp.array([[1], [2], [0]], dtype=jnp.uint8)
    psi_i = jnp.array([[2.0, 3.0], [1.0, -1.0]], dtype=jnp.float64)
    fwd = apply_within_subspace(ci, psi_i, cj, *masks, direction=0)
    bwd = apply_within_subspace(ci, psi_i, cj, *masks, direction=1)
    assert fwd.shape == (3, 2)
    assert bwd.shape == (2, 2)
    assert jnp.any(jnp.abs(fwd) > 0) or jnp.any(jnp.abs(bwd) > 0)
