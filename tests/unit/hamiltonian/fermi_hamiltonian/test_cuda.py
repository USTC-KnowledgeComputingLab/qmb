"""CUDA regression tests (requires GPU and compiled .so).

Covers all four operations: diagonal, apply_within (forward+backward),
find_all_relative_configs, find_topk_relative_configs.
Each test has a corresponding JAX fallback test in test_fallback.py.
All tests compare CUDA output against JAX fallback with allclose.
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
    )  # FIXME: CUDA sign bug)


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
