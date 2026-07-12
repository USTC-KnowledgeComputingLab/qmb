"""Pure JAX fallback implementations of the four Hamiltonian operations.

Used when CUDA .so is not available (CPU, CI, macOS).
All functions are JIT-compatible where fixed shapes are known.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array


def _parity(pc: jax.Array, pm: jax.Array, config: jax.Array, n_qubytes: int) -> jax.Array:
    """Compute JW parity: parity_const XOR popcount(parity_mask & config)."""
    p = pc.astype(jnp.int32) & 1
    for q in range(n_qubytes):
        p ^= jnp.bitwise_count(jnp.uint32(pm[q] & config[q])) & 1
    return p


@jax.jit
def compute_diagonal_within_subspace(
    configs: Array,
    create_mask: Array,
    annihilate_mask: Array,
    flip_mask: Array,
    parity_mask: Array,
    parity_const: Array,
    coef: Array,
) -> Array:
    """Compute diagonal Hamiltonian elements. Only flip_mask==0 terms contribute."""
    batch_size, n_qubytes = configs.shape
    term_count = coef.shape[0]
    psi = jnp.zeros((batch_size, 2), dtype=jnp.float64)

    for t in range(term_count):
        cm = create_mask[t]
        am = annihilate_mask[t]
        fm = flip_mask[t]
        pm_mask = parity_mask[t]
        pc = parity_const[t]
        is_diag = jnp.all(fm == 0)
        applicable = jnp.ones(batch_size, dtype=bool)
        for q in range(n_qubytes):
            applicable &= (configs[:, q] & cm[q]) == 0
            applicable &= (configs[:, q] & am[q]) == am[q]
        applicable &= is_diag
        parity = pc.astype(jnp.int32) & 1
        for q in range(n_qubytes):
            parity ^= jnp.bitwise_count(jnp.uint32(pm_mask[q] & configs[:, q])) & 1
        sign = jnp.where(parity.astype(bool), -1.0, 1.0)
        mask = applicable.astype(jnp.float64)
        psi = psi.at[:, 0].add(mask * sign * coef[t, 0])
        psi = psi.at[:, 1].add(mask * sign * coef[t, 1])
    return psi


@partial(jax.jit, static_argnums=(9,))
def apply_within_subspace(
    configs_i: Array,
    psi_i: Array,
    configs_j: Array,
    create_mask: Array,
    annihilate_mask: Array,
    flip_mask: Array,
    parity_mask: Array,
    parity_const: Array,
    coef: Array,
    direction: int = 0,
) -> Array:
    """Apply H|psi_i> projected onto configs_j subspace."""
    if direction == 0:
        src_c, src_p, dst_c = configs_i, psi_i, configs_j
    else:
        src_c, src_p, dst_c = configs_j, psi_i, configs_i
    batch_size_src, n_qubytes = src_c.shape
    batch_size_dst = dst_c.shape[0]
    term_count = coef.shape[0]
    psi_j = jnp.zeros((batch_size_dst, 2), dtype=jnp.float64)

    for t in range(term_count):
        cm = create_mask[t]
        am = annihilate_mask[t]
        fm = flip_mask[t]
        pm_mask = parity_mask[t]
        pc = parity_const[t]
        for i in range(batch_size_src):
            check_c = jnp.where(direction == 1, src_c[i] ^ fm, src_c[i])
            applicable = True
            for q in range(n_qubytes):
                applicable &= (check_c[q] & cm[q]) == 0
                applicable &= (check_c[q] & am[q]) == am[q]
            new_c = src_c[i] ^ fm
            matches = jnp.all(dst_c == new_c, axis=1)
            matched = jnp.any(matches)
            idx = jnp.argmax(matches)
            parity = _parity(pc, pm_mask, check_c, n_qubytes)
            sign = jnp.where(parity.astype(bool), -1.0, 1.0)
            cf_r = coef[t, 0]
            cf_i = jnp.where(direction == 1, -coef[t, 1], coef[t, 1])
            pr = src_p[i, 0]
            pi_v = src_p[i, 1]
            val_r = sign * (cf_r * pr - cf_i * pi_v)
            val_i = sign * (cf_r * pi_v + cf_i * pr)
            add = applicable & matched
            psi_j = psi_j.at[idx, 0].add(jnp.where(add, val_r, 0.0))
            psi_j = psi_j.at[idx, 1].add(jnp.where(add, val_i, 0.0))
    return psi_j


def find_all_relative_configs(
    configs_i: Array,
    psi_i: Array,
    configs_exclude: Array,
    create_mask: Array,
    annihilate_mask: Array,
    flip_mask: Array,
    parity_mask: Array,
    parity_const: Array,
    coef: Array,
    hash_capacity: int,
) -> tuple[Array, Array, Array]:
    """Enumerate all unique new configs reachable via H, with amplitude accumulation."""
    batch_size, n_qubytes = configs_i.shape
    term_count = coef.shape[0]
    cap = hash_capacity
    keys = jnp.zeros((cap, n_qubytes), dtype=jnp.uint8)
    vals = jnp.zeros((cap, 2), dtype=jnp.float64)
    occupied = jnp.zeros(cap, dtype=bool)
    count = jnp.array(0, dtype=jnp.int32)

    for t in range(term_count):
        cm = create_mask[t]
        am = annihilate_mask[t]
        fm = flip_mask[t]
        pm_mask = parity_mask[t]
        pc = parity_const[t]
        for i in range(batch_size):
            c = configs_i[i]
            applicable = True
            for q in range(n_qubytes):
                if (c[q] & cm[q]) != 0:
                    applicable = False
                    break
                if (c[q] & am[q]) != am[q]:
                    applicable = False
                    break
            if not applicable:
                continue
            new_c = c ^ fm
            excluded = False
            for e in range(configs_exclude.shape[0]):
                if jnp.all(configs_exclude[e] == new_c):
                    excluded = True
                    break
            if excluded:
                continue
            parity = _parity(pc, pm_mask, c, n_qubytes)
            sign = jnp.where(parity.astype(bool), -1.0, 1.0)
            cr = coef[t, 0]
            ci = coef[t, 1]
            pr = psi_i[i, 0]
            pi_v = psi_i[i, 1]
            val_r = sign * (cr * pr - ci * pi_v)
            val_i = sign * (cr * pi_v + ci * pr)
            found = -1
            for s in range(cap):
                if occupied[s] and jnp.all(keys[s] == new_c):
                    found = s
                    break
            if found >= 0:
                vals = vals.at[found, 0].add(val_r)
                vals = vals.at[found, 1].add(val_i)
            elif count < cap:
                keys = keys.at[count].set(new_c)
                vals = vals.at[count, 0].set(val_r)
                vals = vals.at[count, 1].set(val_i)
                occupied = occupied.at[count].set(True)
                count = count + 1
    return keys, vals, count


def find_topk_relative_configs(
    configs_i: Array,
    psi_i: Array,
    count_selected: int,
    configs_exclude: Array,
    create_mask: Array,
    annihilate_mask: Array,
    flip_mask: Array,
    parity_mask: Array,
    parity_const: Array,
    coef: Array,
) -> Array:
    """Select top-K configs by max weight."""
    batch_size, n_qubytes = configs_i.shape
    term_count = coef.shape[0]
    count_selected = count_selected
    cap = count_selected * 2
    keys = jnp.zeros((cap, n_qubytes), dtype=jnp.uint8)
    weights = jnp.zeros(cap, dtype=jnp.float64)
    occupied = jnp.zeros(cap, dtype=bool)
    cnt = jnp.array(0, dtype=jnp.int32)

    for t in range(term_count):
        cm = create_mask[t]
        am = annihilate_mask[t]
        fm = flip_mask[t]
        for i in range(batch_size):
            c = configs_i[i]
            applicable = True
            for q in range(n_qubytes):
                if (c[q] & cm[q]) != 0:
                    applicable = False
                    break
                if (c[q] & am[q]) != am[q]:
                    applicable = False
                    break
            if not applicable:
                continue
            new_c = c ^ fm
            excluded = False
            for e in range(configs_exclude.shape[0]):
                if jnp.all(configs_exclude[e] == new_c):
                    excluded = True
                    break
            if excluded:
                continue
            cr = coef[t, 0]
            ci = coef[t, 1]
            pr = psi_i[i, 0]
            pi_v = psi_i[i, 1]
            weight = (cr * pr - ci * pi_v) ** 2 + (cr * pi_v + ci * pr) ** 2
            found = -1
            for s in range(cap):
                if occupied[s] and jnp.all(keys[s] == new_c):
                    found = s
                    break
            if found >= 0:
                prev = weights[found]
                weights = weights.at[found].set(jnp.maximum(prev, weight))
            elif cnt < cap:
                keys = keys.at[cnt].set(new_c)
                weights = weights.at[cnt].set(weight)
                occupied = occupied.at[cnt].set(True)
                cnt = cnt + 1

    idx = jnp.argsort(weights)[::-1][:count_selected]
    return keys[idx]
