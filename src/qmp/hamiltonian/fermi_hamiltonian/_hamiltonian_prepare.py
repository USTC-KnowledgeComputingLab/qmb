"""Pure Python bit-mask preparation for fermionic Hamiltonians."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array


def prepare(
    hamiltonian: dict[tuple[tuple[int, int], ...], complex],
    n_qubits: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Convert a Hamiltonian dict to bit-mask representation.

    Parameters
    ----------
    hamiltonian : dict
        Keys are tuples of (site_index, kind) where kind=1 for creation,
        kind=0 for annihilation. Values are complex coefficients.
    n_qubits : int
        Number of qubits (orbitals * 2 for fermions).

    Returns
    -------
    tuple[Array, Array, Array, Array, Array, Array]
        (create_mask [T,Q] uint8, annihilate_mask [T,Q] uint8,
         flip_mask [T,Q] uint8, parity_mask [T,Q] uint8,
         parity_const [T] uint8, coef [T,2] float64)
    """
    n_qubytes = (n_qubits + 7) // 8
    terms: list[tuple[list[tuple[int, bool]], complex]] = []
    for key, value in hamiltonian.items():
        ops: list[tuple[int, bool]] = []
        for site, kind in key:
            if kind == 1:
                ops.append((site, True))
            elif kind == 0:
                ops.append((site, False))
            # kind == 2 (identity) silently skipped
        if ops:
            terms.append((ops, value))

    create_mask_list: list[int] = []
    annihilate_mask_list: list[int] = []
    flip_mask_list: list[int] = []
    parity_mask_list: list[int] = []
    parity_const_list: list[int] = []
    coef_list: list[tuple[float, float]] = []

    for ops, coef_val in terms:
        result = _process_term(ops, n_qubits)
        if result is None:
            continue
        cm, am, fm, pm, pc = result
        create_mask_list.append(cm)
        annihilate_mask_list.append(am)
        flip_mask_list.append(fm)
        parity_mask_list.append(pm)
        parity_const_list.append(pc)
        coef_list.append((coef_val.real, coef_val.imag))

    term_count = len(create_mask_list)

    def _to_array(values: list[int]) -> Array:
        arr = jnp.zeros((term_count, n_qubytes), dtype=jnp.uint8)
        for t, val in enumerate(values):
            for q in range(n_qubytes):
                arr = arr.at[t, q].set(jnp.uint8((val >> (q * 8)) & 0xFF))
        return arr

    create_mask = _to_array(create_mask_list)
    annihilate_mask = _to_array(annihilate_mask_list)
    flip_mask = _to_array(flip_mask_list)
    parity_mask = _to_array(parity_mask_list)
    parity_const = jnp.array(parity_const_list, dtype=jnp.uint8)
    reals, imags = zip(*coef_list, strict=True) if coef_list else ([], [])
    coef = jnp.stack([jnp.array(reals, dtype=jnp.float64), jnp.array(imags, dtype=jnp.float64)], axis=1)

    return create_mask, annihilate_mask, flip_mask, parity_mask, parity_const, coef


def _process_term(ops: list[tuple[int, bool]], n_qubits: int) -> tuple[int, int, int, int, int] | None:
    """Process one term's operator sequence.

    Application order = reverse of writing order.
    Returns (create_mask, annihilate_mask, flip_mask, parity_mask, parity_const)
    or None if term is identically zero.
    """
    known = [False] * n_qubits
    initial = [0] * n_qubits
    flip = 0
    p_const = 0
    p_mask = 0

    for s, c in reversed(ops):
        flip_s = (flip >> s) & 1
        target = 0 if c else 1  # create → 0, annihilate → 1
        if known[s]:
            if (initial[s] ^ flip_s) != target:
                return None
        else:
            initial[s] = target
            known[s] = True

        lo = (1 << s) - 1
        known_mask = 0
        for i in range(n_qubits):
            if known[i]:
                known_mask |= 1 << i
        unknown_mask = lo & ~known_mask
        contrib = 0
        for i in range(s):
            if known[i]:
                contrib ^= initial[i] ^ ((flip >> i) & 1)
        p_const ^= contrib & 1
        p_mask ^= unknown_mask

        flip ^= 1 << s

    create_mask = 0
    annihilate_mask = 0
    for i in range(n_qubits):
        if known[i] and initial[i] == 0:
            create_mask |= 1 << i
        if known[i] and initial[i] == 1:
            annihilate_mask |= 1 << i

    return (create_mask, annihilate_mask, int(flip), int(p_mask), p_const)
