"""Tests for Hamiltonian term bit-mask preparation."""

from __future__ import annotations

from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_prepare import prepare


def _h2_hamiltonian() -> dict[tuple[tuple[int, int], ...], complex]:
    """H2 STO-3G Hamiltonian: 4 spin-orbitals."""
    return {
        ((0, 1), (0, 0)): 0.715104 * (-1) + 0j,
    }


def test_prepare_empty() -> None:
    """Empty dict returns zero-length arrays."""
    result = prepare({}, n_qubits=4)
    assert len(result) == 6
    for arr in result:
        assert int(arr.shape[0]) == 0


def test_prepare_identity_skip() -> None:
    """Terms with kind=2 (identity) should be skipped."""
    h = {((0, 2),): 1.0 + 0j}
    result = prepare(h, n_qubits=4)
    assert int(result[0].shape[0]) == 0


def test_prepare_number_operator() -> None:
    """c_0^dag c_0 = n_0: annihilate_mask={0}, flip_mask=0, parity_mask=0."""
    h = {((0, 1), (0, 0)): 1.0 + 0j}  # create at 0, annihilate at 0
    result = prepare(h, n_qubits=8)
    _cm, _am, _fm, _pm, _pc, _coef = result
    assert int(result[0].shape[0]) == 1
    assert int(_am[0, 0]) == 1
    assert int(_cm[0, 0]) == 0
    assert int(_fm[0, 0]) == 0
    assert int(_pm[0, 0]) == 0
    assert int(_pc[0]) == 0


def test_prepare_conflict_zero() -> None:
    """c_0 c_0 should be identically zero (Pauli exclusion)."""
    h = {((0, 0), (0, 0)): 1.0 + 0j}
    result = prepare(h, n_qubits=4)
    assert int(result[0].shape[0]) == 0


def _make_simple_hubbard_2site() -> dict[tuple[tuple[int, int], ...], complex]:
    """2-site spinless Hubbard-like model: single hopping term c_1^dag c_0."""
    return {((1, 1), (0, 0)): -1.0 + 0j}


def test_prepare_hubbard_2site_masks() -> None:
    """c_1^dag c_0: create_mask={1}, annihilate_mask={0}, flip_mask={0,1}."""
    h = _make_simple_hubbard_2site()
    result = prepare(h, n_qubits=4)
    _cm, _am, _fm, _pm, _pc, _coef = result
    assert int(result[0].shape[0]) == 1
    assert int(_cm[0, 0]) & 2 == 2  # bit 1 set
    assert int(_am[0, 0]) & 1 == 1  # bit 0 set
    assert (int(_fm[0, 0]) & 3) == 3  # both bits set


def test_prepare_coef_preserved() -> None:
    """Coefficients should be preserved through prepare."""
    h = {((0, 1), (0, 0)): 3.5 - 2.0j}
    result = prepare(h, n_qubits=4)
    _coef = result[5]
    assert abs(float(_coef[0, 0]) - 3.5) < 1e-10
    assert abs(float(_coef[0, 1]) - (-2.0)) < 1e-10


def test_prepare_four_op_term() -> None:
    """c_3^dag c_1^dag c_5 c_7: 2 creates + 2 annihilates on 8 qubits."""
    ops = ((3, 1), (1, 1), (5, 0), (7, 0))
    h = {ops: 1.0 + 0j}
    result = prepare(h, n_qubits=8)
    _cm, _am, _fm, _pm, _pc, _coef = result
    assert int(result[0].shape[0]) == 1
    cm = int(_cm[0, 0])
    am = int(_am[0, 0])
    assert cm == 10  # bits 1,3
    assert am == 160  # bits 5,7


def test_prepare_parity_mask_jw() -> None:
    """c_2^dag c_0 on 4 qubits: JW parity should depend on qubit 1."""
    h = {((2, 1), (0, 0)): 1.0 + 0j}  # create at 2, annihilate at 0
    result = prepare(h, n_qubits=4)
    _pm = result[3]
    assert int(_pm[0, 0]) == 2  # bit 1 set (parity depends on qubit 1)


def test_prepare_create_mask_h2() -> None:
    """H2 Hamiltonian: verify create_mask for c_0^dag c_0 term."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((0, 1), (0, 0)): 0.715104 * (-1) + 0j}
    result = prepare(h, n_qubits=4)
    _cm = result[0]
    assert int(_cm[0, 0]) == 0  # c_0^dag c_0 has no create bits


def test_prepare_non_aligned_qubits() -> None:
    """n_qubits=10 (not byte-aligned, Q=2). Should produce correct masks."""
    h = {((1, 1), (0, 0)): -1.0 + 0j}
    result = prepare(h, n_qubits=10)
    assert result[0].shape == (1, 2)  # Q = ceil(10/8) = 2
    assert int(result[0][0, 0]) & 2 == 2  # bit 1 in first byte
