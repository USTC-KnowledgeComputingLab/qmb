"""Pure JAX fallback implementations of the four Hamiltonian operations.

Used when CUDA .so is not available (CPU, CI, macOS).
All functions are JIT-compatible: no Python control-flow on traced values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array


def _parity(pc: jax.Array, pm: jax.Array, config: jax.Array, Q: int) -> jax.Array:
    """Compute JW parity: parity_const XOR popcount(parity_mask & config).
    Returns 0 or 1 as a JAX scalar array (JIT-compatible)."""
    p = pc.astype(jnp.int32) & 1
    for q in range(Q):
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
    B, Q = configs.shape
    T = coef.shape[0]
    psi = jnp.zeros((B, 2), dtype=jnp.float64)

    for t in range(T):
        cm = create_mask[t]
        am = annihilate_mask[t]
        fm = flip_mask[t]
        pm_mask = parity_mask[t]
        pc = parity_const[t]

        # check flip_mask == 0 for all qubytes (diagonal condition)
        is_diag = jnp.all(fm == 0)

        # applicability check across all configs
        applicable = jnp.ones(B, dtype=bool)
        for q in range(Q):
            applicable &= (configs[:, q] & cm[q]) == 0
            applicable &= (configs[:, q] & am[q]) == am[q]
        applicable &= is_diag

        # compute parity for all configs
        parity = pc.astype(jnp.int32) & 1
        for q in range(Q):
            parity ^= jnp.bitwise_count(jnp.uint32(pm_mask[q] & configs[:, q])) & 1
        sign = jnp.where(parity.astype(bool), -1.0, 1.0)

        # accumulate: mask non-applicable entries to zero, accumulate
        mask = applicable.astype(jnp.float64)
        psi = psi.at[:, 0].add(mask * sign * coef[t, 0])
        psi = psi.at[:, 1].add(mask * sign * coef[t, 1])

    return psi


@jax.jit
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

    B_src, Q = src_c.shape
    B_dst = dst_c.shape[0]
    T = coef.shape[0]
    psi_j = jnp.zeros((B_dst, 2), dtype=jnp.float64)

    for t in range(T):
        cm = create_mask[t]
        am = annihilate_mask[t]
        fm = flip_mask[t]
        pm_mask = parity_mask[t]
        pc = parity_const[t]

        # backward: check config_i (= config_j XOR flip)
        for i in range(B_src):
            check_c = src_c[i] ^ fm if direction == 1 else src_c[i]
            applicable = True
            for q in range(Q):
                if (check_c[q] & cm[q]) != 0:
                    applicable = False
                    break
                if (check_c[q] & am[q]) != am[q]:
                    applicable = False
                    break
            if not applicable:
                continue

            new_c = src_c[i] ^ fm

            # linear search in dst configs
            idx = -1
            for j in range(B_dst):
                if jnp.all(dst_c[j] == new_c):
                    idx = j
                    break
            if idx < 0:
                continue

            parity = _parity(pc, pm_mask, check_c, Q)
            sign = jnp.where(parity.astype(bool), -1.0, 1.0)
            cf_r = coef[t, 0]
            cf_i = -coef[t, 1] if direction == 1 else coef[t, 1]
            pr = src_p[i, 0]
            pi_v = src_p[i, 1]
            val_r = sign * (cf_r * pr - cf_i * pi_v)
            val_i = sign * (cf_r * pi_v + cf_i * pr)
            psi_j = psi_j.at[idx, 0].add(val_r)
            psi_j = psi_j.at[idx, 1].add(val_i)

    return psi_j


@jax.jit
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
    B, Q = configs_i.shape
    T = coef.shape[0]
    cap = hash_capacity
    keys = jnp.zeros((cap, Q), dtype=jnp.uint8)
    vals = jnp.zeros((cap, 2), dtype=jnp.float64)
    occupied = jnp.zeros(cap, dtype=bool)
    count = jnp.array(0, dtype=jnp.int32)

    for t in range(T):
        cm = create_mask[t]
        am = annihilate_mask[t]
        fm = flip_mask[t]
        pm_mask = parity_mask[t]
        pc = parity_const[t]
        for i in range(B):
            c = configs_i[i]
            applicable = True
            for q in range(Q):
                if (c[q] & cm[q]) != 0:
                    applicable = False
                    break
                if (c[q] & am[q]) != am[q]:
                    applicable = False
                    break
            if not applicable:
                continue
            new_c = c ^ fm
            # exclude check
            excluded = False
            for e in range(configs_exclude.shape[0]):
                if jnp.all(configs_exclude[e] == new_c):
                    excluded = True
                    break
            if excluded:
                continue
            parity = _parity(pc, pm_mask, c, Q)
            sign = jnp.where(parity.astype(bool), -1.0, 1.0)
            cr = coef[t, 0]
            ci = coef[t, 1]
            pr = psi_i[i, 0]
            pi_v = psi_i[i, 1]
            val_r = sign * (cr * pr - ci * pi_v)
            val_i = sign * (cr * pi_v + ci * pr)
            # linear probe
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


@jax.jit
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
    B, Q = configs_i.shape
    T = coef.shape[0]
    K = count_selected
    cap = K * 2
    keys = jnp.zeros((cap, Q), dtype=jnp.uint8)
    weights = jnp.zeros(cap, dtype=jnp.float64)
    occupied = jnp.zeros(cap, dtype=bool)
    cnt = jnp.array(0, dtype=jnp.int32)

    for t in range(T):
        cm = create_mask[t]
        am = annihilate_mask[t]
        fm = flip_mask[t]
        for i in range(B):
            c = configs_i[i]
            applicable = True
            for q in range(Q):
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

    idx = jnp.argsort(weights)[::-1][:K]
    return keys[idx]
