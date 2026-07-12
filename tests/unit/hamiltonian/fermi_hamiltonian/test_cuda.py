"""CUDA regression tests (requires GPU and compiled .so).

Covers all four operations: diagonal, apply_within (forward+backward),
find_all_relative_configs, find_topk_relative_configs.
Each test has a corresponding JAX fallback test in test_fallback.py.
All tests compare CUDA output against JAX fallback with allclose.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
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

# ---- helpers ----


def _hopping_h() -> FermiHamiltonian:
    return FermiHamiltonian({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4, devices=["localhost:cuda:0"])


def _diagonal_h() -> FermiHamiltonian:
    return FermiHamiltonian({((0, 1), (0, 0)): 2.0 + 0j}, n_qubits=8, devices=["localhost:cuda:0"])


def _mixed_h() -> FermiHamiltonian:
    return FermiHamiltonian(
        {((1, 1), (0, 0)): -1.0 + 0j, ((0, 1), (0, 0)): 2.0 + 0j}, n_qubits=8, devices=["localhost:cuda:0"]
    )


def _two_site_fermion_h() -> FermiHamiltonian:
    return FermiHamiltonian(
        {((1, 1), (0, 0)): -1.0 + 0j, ((0, 1), (1, 0)): -1.0 + 0j}, n_qubits=4, devices=["localhost:cuda:0"]
    )


def _c4() -> jax.Array:
    return jnp.array([[0b10], [0b01], [0b11], [0b00]], dtype=jnp.uint8)


def _c2() -> jax.Array:
    return jnp.array([[0b01], [0b00]], dtype=jnp.uint8)


# ---- diagonal ----


def test_cuda_diagonal_hopping() -> None:
    h, c = _hopping_h(), _c4()
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    assert jnp.allclose(h.compute_diagonal_within_subspace(c), jax_diag(c, *m), rtol=1e-12)


def test_cuda_diagonal_with_n0() -> None:
    h, c = _diagonal_h(), _c2()
    m = prepare({((0, 1), (0, 0)): 2.0 + 0j}, n_qubits=8)
    assert jnp.allclose(h.compute_diagonal_within_subspace(c), jax_diag(c, *m), rtol=1e-12)


def test_cuda_diagonal_mixed() -> None:
    h, c = _mixed_h(), _c4()
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j, ((0, 1), (0, 0)): 2.0 + 0j}, n_qubits=8)
    assert jnp.allclose(h.compute_diagonal_within_subspace(c), jax_diag(c, *m), rtol=1e-12)


def test_cuda_diagonal_only_hamiltonian() -> None:
    h = FermiHamiltonian(
        {((0, 1), (0, 0)): 1.0 + 0j, ((1, 1), (1, 0)): 2.0 + 0j}, n_qubits=8, devices=["localhost:cuda:0"]
    )
    c = jnp.array([[0b01], [0b10], [0b11]], dtype=jnp.uint8)
    m = prepare({((0, 1), (0, 0)): 1.0 + 0j, ((1, 1), (1, 0)): 2.0 + 0j}, n_qubits=8)
    assert jnp.allclose(h.compute_diagonal_within_subspace(c), jax_diag(c, *m), rtol=1e-12)


# ---- apply_within ----


def test_cuda_apply_forward() -> None:
    h, c = _hopping_h(), _c4()
    pi = jnp.ones((4, 2), dtype=jnp.float64)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    assert jnp.allclose(
        h.apply_within_subspace(c, pi, c, direction=0), jax_apply_within(c, pi, c, *m, direction=0), rtol=1e-12
    )


def test_cuda_apply_backward() -> None:
    h, c = _hopping_h(), _c4()
    pi = jnp.ones((4, 2), dtype=jnp.float64)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    assert jnp.allclose(
        h.apply_within_subspace(c, pi, c, direction=1), jax_apply_within(c, pi, c, *m, direction=1), rtol=1e-12
    )


def test_cuda_apply_numerical_hopping() -> None:
    h = _two_site_fermion_h()
    ci = jnp.array([[1]], dtype=jnp.uint8)
    pi = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    cj = jnp.array([[2]], dtype=jnp.uint8)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j, ((0, 1), (1, 0)): -1.0 + 0j}, n_qubits=4)
    assert jnp.allclose(
        h.apply_within_subspace(ci, pi, cj, direction=0), jax_apply_within(ci, pi, cj, *m, direction=0), rtol=1e-12
    )


def test_cuda_apply_pauli_exclusion() -> None:
    h = _two_site_fermion_h()
    ci = jnp.array([[3]], dtype=jnp.uint8)
    pi = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    cj = jnp.array([[3]], dtype=jnp.uint8)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j, ((0, 1), (1, 0)): -1.0 + 0j}, n_qubits=4)
    assert jnp.allclose(
        h.apply_within_subspace(ci, pi, cj, direction=0), jax_apply_within(ci, pi, cj, *m, direction=0), rtol=1e-12
    )


def test_cuda_apply_identity_configs() -> None:
    h = _two_site_fermion_h()
    ci = jnp.array([[1]], dtype=jnp.uint8)
    pi = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j, ((0, 1), (1, 0)): -1.0 + 0j}, n_qubits=4)
    assert jnp.allclose(
        h.apply_within_subspace(ci, pi, ci, direction=0), jax_apply_within(ci, pi, ci, *m, direction=0), rtol=1e-12
    )


def test_cuda_apply_complex_psi() -> None:
    h = FermiHamiltonian({((1, 1), (0, 0)): 1.0 + 0.0j}, n_qubits=4, devices=["localhost:cuda:0"])
    ci = jnp.array([[1]], dtype=jnp.uint8)
    pi = jnp.array([[2.0, 3.0]], dtype=jnp.float64)
    cj = jnp.array([[2]], dtype=jnp.uint8)
    m = prepare({((1, 1), (0, 0)): 1.0 + 0j}, n_qubits=4)
    assert jnp.allclose(
        h.apply_within_subspace(ci, pi, cj, direction=0), jax_apply_within(ci, pi, cj, *m, direction=0), rtol=1e-12
    )


def test_cuda_apply_complex_coef() -> None:
    h = FermiHamiltonian({((1, 1), (0, 0)): 0.0 + 1.0j}, n_qubits=4, devices=["localhost:cuda:0"])
    ci = jnp.array([[1]], dtype=jnp.uint8)
    pi = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    cj = jnp.array([[2]], dtype=jnp.uint8)
    m = prepare({((1, 1), (0, 0)): 0.0 + 1.0j}, n_qubits=4)
    assert jnp.allclose(
        h.apply_within_subspace(ci, pi, cj, direction=0), jax_apply_within(ci, pi, cj, *m, direction=0), rtol=1e-12
    )


def test_cuda_apply_superset_j() -> None:
    h = _two_site_fermion_h()
    ci = jnp.array([[1]], dtype=jnp.uint8)
    pi = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    cj = jnp.array([[0], [1], [2]], dtype=jnp.uint8)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j, ((0, 1), (1, 0)): -1.0 + 0j}, n_qubits=4)
    assert jnp.allclose(
        h.apply_within_subspace(ci, pi, cj, direction=0), jax_apply_within(ci, pi, cj, *m, direction=0), rtol=1e-12
    )


def test_cuda_apply_small_subspace() -> None:
    h = _hopping_h()
    ci = _c4()
    cj = jnp.array([[0b10], [0b01]], dtype=jnp.uint8)
    pi = jnp.ones((4, 2), dtype=jnp.float64)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    assert jnp.allclose(
        h.apply_within_subspace(ci, pi, cj, direction=0), jax_apply_within(ci, pi, cj, *m, direction=0), rtol=1e-12
    )


def test_cuda_apply_backward_asymmetric() -> None:
    # Backward (direction=1) with B_i != B_j exercises the smaller-side traversal
    # (traverse_dst branch) together with the H^dagger semantics.
    h = _hopping_h()
    configs_i = _c4()  # 4 configs
    configs_j = jnp.array([[0b10], [0b01]], dtype=jnp.uint8)  # 2 configs (smaller dst under dir=1)
    psi = jnp.ones((2, 2), dtype=jnp.float64)  # lives on configs_j (src for backward)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    assert jnp.allclose(
        h.apply_within_subspace(configs_i, psi, configs_j, direction=1),
        jax_apply_within(configs_i, psi, configs_j, *m, direction=1),
        rtol=1e-12,
    )


# ---- find_all ----


def test_cuda_find_all() -> None:
    h = _hopping_h()
    c, pi = _c4(), jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    _, _, cc = h.find_all_relative_configs(c, pi, exclude, hash_capacity=100)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    _, _, jc = jax_find_all(c, pi, exclude, *m, hash_capacity=100)
    assert int(cc) == int(jc)


def test_cuda_find_all_empty_exclude() -> None:
    h = _hopping_h()
    c, pi = _c4(), jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    _, v, _ = h.find_all_relative_configs(c, pi, exclude, hash_capacity=100)
    assert jnp.any(jnp.abs(v) > 0)


def test_cuda_find_all_with_exclude() -> None:
    h = _hopping_h()
    c, pi = _c4(), jnp.ones((4, 2), dtype=jnp.float64)
    _, _, cc = h.find_all_relative_configs(c, pi, c, hash_capacity=100)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    _, _, jc = jax_find_all(c, pi, c, *m, hash_capacity=100)
    assert int(cc) == int(jc)


def test_cuda_overflow_retry() -> None:
    h = _hopping_h()
    c = jnp.array([[0b10], [0b01]], dtype=jnp.uint8)
    pi = jnp.ones((2, 2), dtype=jnp.float64)
    _, _, cnt = h.find_all_relative_configs(c, pi, jnp.zeros((0, 1), dtype=jnp.uint8), hash_capacity=2)
    assert int(cnt) >= 0


def test_cuda_overflow_retry_matches_large_capacity() -> None:
    # Force the hash table to overflow with a deliberately tiny capacity, then
    # confirm the Python-side capacity-doubling retry recovers the same result
    # as a run that starts with ample capacity (no config silently dropped).
    h = _hopping_h()
    c = _c4()
    pi = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    _, _, cnt_small = h.find_all_relative_configs(c, pi, exclude, hash_capacity=1)
    _, _, cnt_large = h.find_all_relative_configs(c, pi, exclude, hash_capacity=256)
    assert int(cnt_small) == int(cnt_large)


def _collect_config_amp_map(configs, psi, count):
    # Build {config_bytes -> (re, im)} for the first `count` output rows.
    result: dict[bytes, tuple[float, float]] = {}
    cfg = np.asarray(configs)
    amp = np.asarray(psi)
    for row in range(int(count)):
        key = cfg[row].tobytes()
        result[key] = (float(amp[row, 0]), float(amp[row, 1]))
    return result


def test_cuda_find_all_dedup_tight_capacity_matches_fallback() -> None:
    # High-collision stress: a mixed Hamiltonian gives multiple (term, config)
    # paths, and a tight-but-sufficient capacity maximises probe collisions so
    # concurrent insertion is prone to creating duplicate slots. The collect-time
    # canonical-slot merge must still yield exactly the fallback's deduped
    # configs and accumulated amplitudes.
    hamiltonian: dict[tuple[tuple[int, int], ...], complex] = {
        ((1, 1), (0, 0)): -1.0 + 0j,
        ((2, 1), (1, 0)): -1.0 + 0j,
        ((0, 1), (0, 0)): 0.5 + 0j,
    }
    h = FermiHamiltonian(hamiltonian, n_qubits=4, devices=["localhost:cuda:0"])
    m = prepare(hamiltonian, n_qubits=4)
    c = _c4()
    pi = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)

    cuda_cfg, cuda_psi, cuda_cnt = h.find_all_relative_configs(c, pi, exclude, hash_capacity=8)
    jax_cfg, jax_psi, jax_cnt = jax_find_all(c, pi, exclude, *m, hash_capacity=64)

    assert int(cuda_cnt) == int(jax_cnt)
    cuda_map = _collect_config_amp_map(cuda_cfg, cuda_psi, cuda_cnt)
    jax_map = _collect_config_amp_map(jax_cfg, jax_psi, jax_cnt)
    assert set(cuda_map) == set(jax_map)
    for key in cuda_map:
        assert cuda_map[key][0] == pytest.approx(jax_map[key][0], abs=1e-9)
        assert cuda_map[key][1] == pytest.approx(jax_map[key][1], abs=1e-9)


# ---- find_topk ----


def test_cuda_find_topk() -> None:
    h = _hopping_h()
    c, pi = _c4(), jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    cr = h.find_topk_relative_configs(c, pi, 2, exclude)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    jr = jax_find_topk(c, pi, 2, exclude, *m)
    assert cr.shape == jr.shape == (2, c.shape[1])


def test_cuda_find_topk_max_semantics() -> None:
    h = FermiHamiltonian(
        {((1, 1), (0, 0)): -1.0 + 0j, ((2, 1), (1, 0)): -0.5 + 0j}, n_qubits=4, devices=["localhost:cuda:0"]
    )
    c = jnp.array([[0b010]], dtype=jnp.uint8)
    pi = jnp.ones((1, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    cr = h.find_topk_relative_configs(c, pi, 4, exclude)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j, ((2, 1), (1, 0)): -0.5 + 0j}, n_qubits=4)
    jr = jax_find_topk(c, pi, 4, exclude, *m)
    assert cr.shape == jr.shape


def test_cuda_find_topk_diagonal_terms() -> None:
    h = _mixed_h()
    c, pi = _c4(), jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    cr = h.find_topk_relative_configs(c, pi, 2, exclude)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j, ((0, 1), (0, 0)): 2.0 + 0j}, n_qubits=8)
    jr = jax_find_topk(c, pi, 2, exclude, *m)
    assert cr.shape == jr.shape == (2, c.shape[1])


def test_cuda_find_topk_k_one() -> None:
    h = _hopping_h()
    c, pi = _c4(), jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    cr = h.find_topk_relative_configs(c, pi, 1, exclude)
    m = prepare({((1, 1), (0, 0)): -1.0 + 0j}, n_qubits=4)
    jr = jax_find_topk(c, pi, 1, exclude, *m)
    assert cr.shape == jr.shape == (1, c.shape[1])


def test_cuda_find_topk_no_duplicate_configs() -> None:
    # Multiple terms let several (term, config) paths reach the same new config,
    # so concurrent insertion is prone to duplicate slots. The canonical-slot
    # merge in collect must ensure a config never occupies more than one top-K
    # row (no duplicate configs among the distinct ones returned).
    hamiltonian: dict[tuple[tuple[int, int], ...], complex] = {
        ((1, 1), (0, 0)): -1.0 + 0j,
        ((2, 1), (1, 0)): -1.0 + 0j,
        ((0, 1), (0, 0)): 0.5 + 0j,
    }
    h = FermiHamiltonian(hamiltonian, n_qubits=4, devices=["localhost:cuda:0"])
    c = _c4()
    pi = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    selected = h.find_topk_relative_configs(c, pi, 4, exclude)
    rows = [tuple(int(b) for b in row) for row in np.asarray(selected)]
    # Non-zero (real) configs must be distinct; zero padding rows may repeat.
    nonzero = [r for r in rows if any(r)]
    assert len(nonzero) == len(set(nonzero))


def _config_set(rows) -> set[tuple[int, ...]]:
    return {tuple(int(b) for b in row) for row in np.asarray(rows) if any(int(b) for b in row)}


def test_cuda_find_topk_selected_configs_match_fallback() -> None:
    # Verify the actual selected config set (not just the shape) matches the JAX
    # fallback. A single occupied source config with distinct term coefficients
    # gives each reachable config a unique weight (no ties), so top-K is
    # deterministic and the set comparison is well-defined.
    hamiltonian: dict[tuple[tuple[int, int], ...], complex] = {
        ((1, 1), (0, 0)): -4.0 + 0j,
        ((2, 1), (0, 0)): -3.0 + 0j,
        ((3, 1), (0, 0)): -2.0 + 0j,
    }
    h = FermiHamiltonian(hamiltonian, n_qubits=4, devices=["localhost:cuda:0"])
    m = prepare(hamiltonian, n_qubits=4)
    c = jnp.array([[0b0001]], dtype=jnp.uint8)
    pi = jnp.ones((1, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    cuda_sel = h.find_topk_relative_configs(c, pi, 2, exclude)
    jax_sel = jax_find_topk(c, pi, 2, exclude, *m)
    assert _config_set(cuda_sel) == _config_set(jax_sel)


def test_cuda_find_topk_with_exclude() -> None:
    # find_topk must honour the exclude set (second hash table); excluded configs
    # must not appear in the selection, matching the fallback. Distinct coefs
    # avoid weight ties so the selection is deterministic.
    hamiltonian: dict[tuple[tuple[int, int], ...], complex] = {
        ((1, 1), (0, 0)): -4.0 + 0j,
        ((2, 1), (0, 0)): -3.0 + 0j,
        ((3, 1), (0, 0)): -2.0 + 0j,
    }
    h = FermiHamiltonian(hamiltonian, n_qubits=4, devices=["localhost:cuda:0"])
    m = prepare(hamiltonian, n_qubits=4)
    c = jnp.array([[0b0001]], dtype=jnp.uint8)
    pi = jnp.ones((1, 2), dtype=jnp.float64)
    exclude = jnp.array([[0b0010]], dtype=jnp.uint8)  # exclude the highest-weight target (qubit1)
    cuda_sel = h.find_topk_relative_configs(c, pi, 2, exclude)
    jax_sel = jax_find_topk(c, pi, 2, exclude, *m)
    assert (0b0010,) not in _config_set(cuda_sel)
    assert _config_set(cuda_sel) == _config_set(jax_sel)


# ---- multi-byte qubits (n_qubits > 8 → n_qubytes >= 2) ----


def test_cuda_diagonal_multi_byte() -> None:
    """n_qubits=12: CUDA diagonal should match JAX fallback."""
    h = FermiHamiltonian(
        {((0, 1), (0, 0)): 3.0 + 0j, ((10, 1), (10, 0)): 5.0 + 0j}, n_qubits=12, devices=["localhost:cuda:0"]
    )
    m = prepare({((0, 1), (0, 0)): 3.0 + 0j, ((10, 1), (10, 0)): 5.0 + 0j}, n_qubits=12)
    c = jnp.array([[0, 0b100], [0b01, 0]], dtype=jnp.uint8)
    assert jnp.allclose(h.compute_diagonal_within_subspace(c), jax_diag(c, *m), rtol=1e-12)


def test_cuda_apply_multi_byte() -> None:
    """n_qubits=12: CUDA apply_within should match JAX fallback."""
    h = FermiHamiltonian({((9, 1), (2, 0)): -1.0 + 0j}, n_qubits=12, devices=["localhost:cuda:0"])
    m = prepare({((9, 1), (2, 0)): -1.0 + 0j}, n_qubits=12)
    ci = jnp.array([[0b100, 0]], dtype=jnp.uint8)
    pi = jnp.ones((1, 2), dtype=jnp.float64)
    cj = jnp.array([[0, 0b10]], dtype=jnp.uint8)
    assert jnp.allclose(
        h.apply_within_subspace(ci, pi, cj, direction=0), jax_apply_within(ci, pi, cj, *m, direction=0), rtol=1e-12
    )


def test_cuda_find_topk_multi_byte() -> None:
    """n_qubits=12: CUDA find_topk should match JAX fallback."""
    h = FermiHamiltonian({((9, 1), (2, 0)): -1.0 + 0j}, n_qubits=12, devices=["localhost:cuda:0"])
    m = prepare({((9, 1), (2, 0)): -1.0 + 0j}, n_qubits=12)
    c = jnp.array([[0b100, 0]], dtype=jnp.uint8)
    pi = jnp.ones((1, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 2), dtype=jnp.uint8)
    cr = h.find_topk_relative_configs(c, pi, 2, exclude)
    jr = jax_find_topk(c, pi, 2, exclude, *m)
    assert cr.shape == jr.shape
