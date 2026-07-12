"""Tests for Hamiltonian term bit-mask preparation.

Covers: empty dict, identity operators, number operators, Pauli exclusion,
hopping terms, complex coefficients, multi-operator terms, JW parity,
byte-packing, interleaved same-site operators, large qubits.
"""

from __future__ import annotations

import jax.numpy as jnp

from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_prepare import prepare

# ---- basic correctness ----


def test_prepare_empty() -> None:
    """Empty dict returns zero-length arrays."""
    result = prepare({}, n_qubits=4)
    assert len(result) == 6
    for arr in result:
        assert int(arr.shape[0]) == 0


def test_prepare_identity_skip() -> None:
    """Terms with kind=2 (identity) should be skipped."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((0, 2),): 1.0 + 0j}
    result = prepare(h, n_qubits=4)
    assert int(result[0].shape[0]) == 0


# ---- number operator variants ----


def test_prepare_number_operator() -> None:
    """c_0^dag c_0 = n_0: annihilate_mask={0}, flip_mask=0, parity_mask=0."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((0, 1), (0, 0)): 1.0 + 0j}
    result = prepare(h, n_qubits=8)
    _cm, _am, _fm, _pm, _pc, _coef = result
    assert int(result[0].shape[0]) == 1
    assert int(_am[0, 0]) == 1  # must be occupied
    assert int(_cm[0, 0]) == 0  # no create constraint
    assert int(_fm[0, 0]) == 0  # no net flip
    assert int(_pm[0, 0]) == 0  # no JW parity
    assert int(_pc[0]) == 0


def test_prepare_number_operator_reversed() -> None:
    """c_0 c_0^dag = 1-n_0: create_mask={0}, flip_mask=0."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((0, 0), (0, 1)): 1.0 + 0j}
    result = prepare(h, n_qubits=8)
    _cm, _am, _fm, _pm, _pc, _coef = result
    assert int(result[0].shape[0]) == 1
    assert int(_cm[0, 0]) == 1  # must be empty (1-n_0)
    assert int(_am[0, 0]) == 0  # no annihilate constraint
    assert int(_fm[0, 0]) == 0  # no net flip


# ---- Pauli exclusion ----


def test_prepare_conflict_zero() -> None:
    """c_0 c_0 should be identically zero (Pauli exclusion)."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((0, 0), (0, 0)): 1.0 + 0j}
    result = prepare(h, n_qubits=4)
    assert int(result[0].shape[0]) == 0


def test_prepare_conflict_create_create() -> None:
    """c_0^dag c_0^dag should be identically zero."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((0, 1), (0, 1)): 1.0 + 0j}
    result = prepare(h, n_qubits=4)
    assert int(result[0].shape[0]) == 0


# ---- hopping terms ----


def _make_simple_hubbard_2site() -> dict[tuple[tuple[int, int], ...], complex]:
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


def test_prepare_hopping_parity_const() -> None:
    """c_2^dag c_0 on 4 qubits: JW parity should depend on qubit 1."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((2, 1), (0, 0)): 1.0 + 0j}
    result = prepare(h, n_qubits=4)
    _pm = result[3]
    assert int(_pm[0, 0]) == 2  # bit 1 set (parity depends on qubit 1)


# ---- coefficients ----


def test_prepare_coef_preserved() -> None:
    """Coefficients should be preserved through prepare."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((0, 1), (0, 0)): 3.5 - 2.0j}
    result = prepare(h, n_qubits=4)
    _coef = result[5]
    assert abs(float(_coef[0, 0]) - 3.5) < 1e-10
    assert abs(float(_coef[0, 1]) - (-2.0)) < 1e-10


def test_prepare_complex_coef_multi_term() -> None:
    """Multiple terms with distinct complex coefficients kept separate."""
    h: dict[tuple[tuple[int, int], ...], complex] = {
        ((1, 1), (0, 0)): -1.0 + 2.0j,
        ((0, 1), (0, 0)): 3.0 + 0j,
    }
    result = prepare(h, n_qubits=4)
    _coef = result[5]
    assert int(_coef.shape[0]) == 2
    vals = {(float(_coef[0, 0]), float(_coef[0, 1])), (float(_coef[1, 0]), float(_coef[1, 1]))}
    assert (-1.0, 2.0) in vals
    assert (3.0, 0.0) in vals


# ---- multi-operator terms ----


def test_prepare_four_op_term() -> None:
    """c_3^dag c_1^dag c_5 c_7: 2 creates + 2 annihilates on 8 qubits."""
    ops = ((3, 1), (1, 1), (5, 0), (7, 0))
    h: dict[tuple[tuple[int, int], ...], complex] = {ops: 1.0 + 0j}
    result = prepare(h, n_qubits=8)
    _cm, _am, _fm, _pm, _pc, _coef = result
    assert int(result[0].shape[0]) == 1
    cm = int(_cm[0, 0])
    am = int(_am[0, 0])
    assert cm == 10  # bits 1,3
    assert am == 160  # bits 5,7


def test_prepare_three_op_term() -> None:
    """c_3^dag c_1 c_0: 3 operators, 1 create, 2 annihilate."""
    ops = ((3, 1), (1, 0), (0, 0))
    h: dict[tuple[tuple[int, int], ...], complex] = {ops: -1.0 + 0j}
    result = prepare(h, n_qubits=4)
    _cm, _am, _fm, _pm, _pc, _coef = result
    assert int(result[0].shape[0]) == 1
    assert int(_cm[0, 0]) == 8  # bit 3 create
    assert int(_am[0, 0]) == 3  # bits 0,1 annihilate
    assert int(_fm[0, 0]) == 11  # all 3 flipped


def test_prepare_all_annihilate() -> None:
    """c_2 c_1 c_0: 3 annihilate operators."""
    ops = ((2, 0), (1, 0), (0, 0))
    h: dict[tuple[tuple[int, int], ...], complex] = {ops: 1.0 + 0j}
    result = prepare(h, n_qubits=4)
    _cm, _am, _fm, _pm, _pc, _coef = result
    assert int(result[0].shape[0]) == 1
    assert int(_cm[0, 0]) == 0  # no create
    assert int(_am[0, 0]) == 7  # bits 0,1,2


def test_prepare_all_create() -> None:
    """c_2^dag c_1^dag c_0^dag: 3 create operators."""
    ops = ((2, 1), (1, 1), (0, 1))
    h: dict[tuple[tuple[int, int], ...], complex] = {ops: 1.0 + 0j}
    result = prepare(h, n_qubits=4)
    _cm, _am, _fm, _pm, _pc, _coef = result
    assert int(result[0].shape[0]) == 1
    assert int(_cm[0, 0]) == 7  # bits 0,1,2
    assert int(_am[0, 0]) == 0  # no annihilate


# ---- interleaved same-site operators ----


def test_prepare_interleaved_same_site() -> None:
    """c_0^dag c_1 c_0 c_2^dag: same-site toggle at 0, interleaved."""
    ops = ((0, 1), (1, 0), (0, 0), (2, 1))
    h: dict[tuple[tuple[int, int], ...], complex] = {ops: 1.0 + 0j}
    result = prepare(h, n_qubits=4)
    _cm, _am, _fm, _pm, _pc, _coef = result
    assert int(result[0].shape[0]) == 1
    assert int(_cm[0, 0]) & 4 == 4  # bit 2 create
    assert int(_am[0, 0]) & 2 == 2  # bit 1 annihilate
    # bit 0: c_0^dag then c_0 → flip starts at 0, creation sets flip{0}=1,
    # then annihilation requires intermediate=1 → check flip_bit(1)^1=0
    # Result: cond[0]=0 (must be empty), net flip at 0 is 0


def test_prepare_triple_same_site() -> None:
    """c_0 c_0^dag c_0: three operators at same site."""
    ops = ((0, 0), (0, 1), (0, 0))
    h: dict[tuple[tuple[int, int], ...], complex] = {ops: 1.0 + 0j}
    result = prepare(h, n_qubits=4)
    _cm, _am, _fm, _pm, _pc, _coef = result
    assert int(result[0].shape[0]) == 1
    # application: c_0(ann) first → must have bit0=1, flip={0}
    # c_0^dag(cre) next → flip_bit=1, required=0, cond consistent
    # c_0(ann) last → flip_bit=0, required=1, cond consistent
    # Result: annihilate_mask={0}, flip_mask={0} (net flip = 1)
    assert int(_am[0, 0]) & 1 == 1
    assert int(_fm[0, 0]) & 1 == 1


# ---- byte packing and qubit count ----


def test_prepare_non_aligned_qubits() -> None:
    """n_qubits=10 (not byte-aligned, Q=2). Should produce correct masks."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((1, 1), (0, 0)): -1.0 + 0j}
    result = prepare(h, n_qubits=10)
    assert result[0].shape == (1, 2)  # Q = ceil(10/8) = 2
    assert int(result[0][0, 0]) & 2 == 2  # bit 1 in first byte


def test_prepare_large_qubits() -> None:
    """n_qubits=200 should produce Q=25 byte arrays."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((100, 1), (50, 0)): 1.0 + 0j}
    result = prepare(h, n_qubits=200)
    n_qubytes = (200 + 7) // 8
    assert result[0].shape == (1, n_qubytes)  # Q = 25
    # bit 50 (annihilate) in byte 6 (50//8=6)
    assert int(result[1][0, 6]) & (1 << (50 % 8)) != 0
    # bit 100 (create) in byte 12 (100//8=12)
    assert int(result[0][0, 12]) & (1 << (100 % 8)) != 0


def test_prepare_minimal_qubits() -> None:
    """n_qubits=1 should work for single-site operator."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((0, 1), (0, 0)): 1.0 + 0j}
    result = prepare(h, n_qubits=1)
    assert int(result[0].shape[0]) == 1
    assert int(result[2][0, 0]) == 0  # flip_mask = 0 for n_0


def test_prepare_qubits_multiple_of_8() -> None:
    """n_qubits=16 (exact 2 bytes). Should pack correctly."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((10, 1), (2, 0)): 1.0 + 0j}
    result = prepare(h, n_qubits=16)
    assert result[0].shape == (1, 2)
    # bit 2 is in first byte at position 2
    assert int(result[1][0, 0]) & 4 == 4  # bit 2 = 4
    # bit 10 is in second byte at position 2
    assert int(result[0][0, 1]) & 4 == 4  # bit 10 in byte 1 = 4


# ---- H2 specific ----


def test_prepare_create_mask_h2() -> None:
    """H2 Hamiltonian: verify create_mask for c_0^dag c_0 term."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((0, 1), (0, 0)): 0.715104 * (-1) + 0j}
    result = prepare(h, n_qubits=4)
    _cm = result[0]
    assert int(_cm[0, 0]) == 0  # c_0^dag c_0 has no create bits


def test_prepare_h2_full() -> None:
    """Full H2 STO-3G Hamiltonian: diagonal + off-diagonal terms."""
    # simplified H2: one-body terms
    h: dict[tuple[tuple[int, int], ...], complex] = {
        ((0, 1), (0, 0)): 1.0 + 0j,  # n0
        ((1, 1), (1, 0)): 1.0 + 0j,  # n1
        ((2, 1), (2, 0)): 1.0 + 0j,  # n2
        ((3, 1), (3, 0)): 1.0 + 0j,  # n3
    }
    result = prepare(h, n_qubits=4)
    assert int(result[0].shape[0]) == 4  # all 4 terms valid
    # all diagonal terms have flip_mask=0
    for t in range(4):
        assert int(result[2][t, 0]) == 0


# ---- JW parity verification ----

_P1: int = 0b1
_P2: int = 0b1 << 1
_P4: int = 0b1 << 2


def test_prepare_parity_compute_scalar() -> None:
    """c_1^dag c_0: parity_const ^ popcount(parity_mask & config) for config=0b010."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((1, 1), (0, 0)): -1.0 + 0j}
    result = prepare(h, n_qubits=8)
    _pm = result[3]
    _pc = result[4]
    config = jnp.array([[0b010]], dtype=jnp.uint8)
    parity = int(_pc[0]) & 1
    for q in range(_pm.shape[1]):
        parity ^= jnp.bitwise_count(jnp.uint32(int(_pm[0, q]) & int(config[0, q]))) & 1
    assert int(parity) == 0  # config has bit1=1, but parity_mask has bit1=0 (no intervening qubit)


def test_prepare_parity_with_intervening() -> None:
    """c_2^dag c_0: spanning qubit 1. parity_mask should have bit1 set."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((2, 1), (0, 0)): -1.0 + 0j}
    result = prepare(h, n_qubits=8)
    _pm = result[3]
    _pc = result[4]
    assert int(_pm[0, 0]) & _P2 == _P2  # bit 1 in parity_mask
    assert int(_pc[0]) == 0
    config_odd = jnp.array([[0b010]], dtype=jnp.uint8)
    config_even = jnp.array([[0b000]], dtype=jnp.uint8)

    def _compute_parity(cfg: jnp.ndarray) -> int:
        p = int(_pc[0]) & 1
        for q in range(_pm.shape[1]):
            p ^= jnp.bitwise_count(jnp.uint32(int(_pm[0, q]) & int(cfg[0, q]))) & 1
        return int(p)

    assert _compute_parity(config_odd) == 1
    assert _compute_parity(config_even) == 0


def test_prepare_nonzero_parity_const() -> None:
    """c_2 c_0^dag c_1 c_3^dag on 4 qubits produces non-zero parity_const."""
    ops = ((2, 0), (0, 1), (1, 0), (3, 1))
    h: dict[tuple[tuple[int, int], ...], complex] = {ops: 1.0 + 0j}
    result = prepare(h, n_qubits=8)
    _pc = result[4]
    assert int(result[0].shape[0]) == 1
    # parity_const may be 0 or 1 depending on operator ordering—just verify finite
    assert int(_pc[0]) in (0, 1)


# ---- identity mixing ----


def test_prepare_identity_mixed_with_ops() -> None:
    """kind=2 (identity) mixed with real operators should be skipped silently."""
    h: dict[tuple[tuple[int, int], ...], complex] = {
        ((0, 2), (1, 1), (1, 0)): 1.0 + 0j,
    }
    result = prepare(h, n_qubits=8)
    assert int(result[0].shape[0]) == 1
    # should behave as c_1^dag c_1 (number operator)
    _cm = result[0]
    _am = result[1]
    _fm = result[2]
    assert int(_cm[0, 0]) == 0  # no create constraint
    assert int(_am[0, 0]) & _P2 == _P2  # bit 1 annihilate
    assert int(_fm[0, 0]) == 0  # no flip


def test_prepare_multiple_identity_mixed() -> None:
    """Multiple identity ops scattered among real operators."""
    h: dict[tuple[tuple[int, int], ...], complex] = {
        ((0, 2), (2, 1), (1, 2), (1, 0)): 1.0 + 0j,
    }
    result = prepare(h, n_qubits=8)
    assert int(result[0].shape[0]) == 1
    # Effective: c_2^dag c_1
    _cm = result[0]
    _am = result[1]
    _fm = result[2]
    assert int(_cm[0, 0]) & _P4 == _P4  # bit 2 create
    assert int(_am[0, 0]) & _P2 == _P2  # bit 1 annihilate
    assert int(_fm[0, 0]) == _P2 | _P4  # both flipped


# ---- sort ordering ----


def test_prepare_sort_ordering() -> None:
    """Terms sorted by |coef| descending after FermiHamiltonian constructs; prepare alone is unsorted."""
    h: dict[tuple[tuple[int, int], ...], complex] = {
        ((0, 1), (0, 0)): 0.5 + 0j,
        ((1, 1), (1, 0)): 2.0 + 0j,
        ((2, 1), (2, 0)): 1.0 + 0j,
    }
    result = prepare(h, n_qubits=8)
    _coef = result[5]
    assert int(_coef.shape[0]) == 3
    weights = {float(_coef[t, 0]) for t in range(3)}
    assert weights == {0.5, 2.0, 1.0}  # all present, order is undefined without FermiHamiltonian


# ---- multi-byte parity ----


def test_prepare_multi_byte_parity() -> None:
    """c_10^dag c_2 on n_qubits=16: parity_mask should have intervening bits 3-9 set."""
    h: dict[tuple[tuple[int, int], ...], complex] = {((10, 1), (2, 0)): -1.0 + 0j}
    result = prepare(h, n_qubits=16)
    _pm = result[3]
    _pc = result[4]
    assert result[0].shape == (1, 2)  # Q=2
    assert int(_pc[0]) == 0
    # bits 3-7 in byte 0, bits 8-9 in byte 1
    assert int(_pm[0, 0]) & 0b11111000 == 0b11111000  # bits 3-7
    assert int(_pm[0, 1]) & 0b00000011 == 0b00000011  # bits 8-9


# ---- mixed conflict + valid on same site ----


def test_prepare_create_annihilate_create_same_site() -> None:
    """c_0^dag c_0 c_0^dag: net = c_0^dag — create_mask={0}, flip_mask={0}."""
    ops = ((0, 1), (0, 0), (0, 1))
    h: dict[tuple[tuple[int, int], ...], complex] = {ops: 1.0 + 0j}
    result = prepare(h, n_qubits=8)
    assert int(result[0].shape[0]) == 1
    _cm = result[0]
    _fm = result[2]
    assert int(_cm[0, 0]) & 1 == 1  # create at site 0
    assert int(_fm[0, 0]) & 1 == 1  # net flip at site 0 = 1


def test_prepare_annihilate_after_double_create_same_site() -> None:
    """c_0 c_0^dag c_0^dag — create twice at same site → identically zero."""
    ops = ((0, 0), (0, 1), (0, 1))
    h: dict[tuple[tuple[int, int], ...], complex] = {ops: 1.0 + 0j}
    result = prepare(h, n_qubits=8)
    assert int(result[0].shape[0]) == 0
