"""Pure JAX fallback implementations of the four Hamiltonian operations.

Used when CUDA .so is not available (CPU, CI, macOS).
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.lax as lax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array


def _parity(pc: jax.Array, pm: jax.Array, config: jax.Array, n_qubytes: int) -> jax.Array:
    """Compute JW parity: parity_const XOR popcount(parity_mask & config)."""
    p = pc.astype(jnp.int32) & 1
    for q in range(n_qubytes):
        p ^= jnp.bitwise_count(jnp.uint32(pm[q] & config[q])) & 1
    return p


def _check_applicable(c: Array, cm: Array, am: Array, n_qubytes: int) -> Array:
    applicable = jnp.ones((), dtype=bool)
    for q in range(n_qubytes):
        applicable &= (c[q] & cm[q]) == 0
        applicable &= (c[q] & am[q]) == am[q]
    return applicable


def _check_excluded(new_c: Array, exclude: Array) -> Array:
    exclude_size = exclude.shape[0]
    excluded = jnp.zeros((), dtype=bool)
    if exclude_size == 0:
        return excluded

    def _body(e: int, carry: Array) -> Array:
        return carry | jnp.all(exclude[e] == new_c)

    return lax.fori_loop(0, exclude_size, _body, excluded)


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
            applicable = _check_applicable(check_c, cm, am, n_qubytes)
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


# ---- JIT-compatible hash table via fori_loop + lax.cond ----


@partial(jax.jit, static_argnums=(9,))
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
    init_keys = jnp.zeros((cap, n_qubytes), dtype=jnp.uint8)
    init_vals = jnp.zeros((cap, 2), dtype=jnp.float64)
    init_occ = jnp.zeros(cap, dtype=bool)
    init_cnt = jnp.array(0, dtype=jnp.int32)

    def _term_body(t: int, state: tuple) -> tuple:
        keys, vals, occ, cnt = state
        cm = create_mask[t]
        am = annihilate_mask[t]
        fm = flip_mask[t]
        pm_mask = parity_mask[t]
        pc = parity_const[t]

        def _config_body(i: int, state: tuple) -> tuple:
            keys, vals, occ, cnt = state
            c = configs_i[i]
            applicable = _check_applicable(c, cm, am, n_qubytes)
            new_c = c ^ fm
            excluded = _check_excluded(new_c, configs_exclude)
            processed = applicable & (~excluded)

            def _do_process(_: Array) -> tuple:
                parity = _parity(pc, pm_mask, c, n_qubytes)
                sign = jnp.where(parity.astype(bool), -1.0, 1.0)
                cr = coef[t, 0]
                ci = coef[t, 1]
                pr = psi_i[i, 0]
                pi_v = psi_i[i, 1]
                val_r = sign * (cr * pr - ci * pi_v)
                val_i = sign * (cr * pi_v + ci * pr)

                def _probe_body(s: int, probe_state: tuple) -> tuple:
                    _keys, _vals, _occ, _cnt, _new_c, _vr, _vi, _matched = probe_state
                    is_match = _occ[s] & jnp.all(_keys[s] == _new_c)

                    def _found() -> tuple:
                        return (
                            _keys,
                            _vals.at[s, 0].add(_vr).at[s, 1].add(_vi),
                            _occ,
                            _cnt,
                            _new_c,
                            _vr,
                            _vi,
                            jnp.ones((), dtype=bool),
                        )

                    def _not_found() -> tuple:
                        return (_keys, _vals, _occ, _cnt, _new_c, _vr, _vi, _matched)

                    return lax.cond(is_match, _found, _not_found)

                new_keys, new_vals, new_occ, new_cnt, _, _, _, matched = lax.fori_loop(
                    0, cap, _probe_body, (keys, vals, occ, cnt, new_c, val_r, val_i, jnp.zeros((), dtype=bool))
                )

                def _insert_empty() -> tuple:
                    return (
                        new_keys.at[new_cnt].set(new_c),
                        new_vals.at[new_cnt, 0].set(val_r).at[new_cnt, 1].set(val_i),
                        new_occ.at[new_cnt].set(True),
                        new_cnt + 1,
                    )

                def _keep() -> tuple:
                    return (new_keys, new_vals, new_occ, new_cnt)

                return lax.cond(
                    (~matched) & (new_cnt < cap),
                    _insert_empty,
                    _keep,
                )

            return lax.cond(processed, _do_process, lambda _: (keys, vals, occ, cnt), None)

        return lax.fori_loop(0, batch_size, _config_body, (keys, vals, occ, cnt))

    final_keys, final_vals, _, final_cnt = lax.fori_loop(
        0, term_count, _term_body, (init_keys, init_vals, init_occ, init_cnt)
    )
    return final_keys, final_vals, final_cnt


@partial(jax.jit, static_argnums=(2,))
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
    cap = count_selected * 2
    init_keys = jnp.zeros((cap, n_qubytes), dtype=jnp.uint8)
    init_weights = jnp.zeros(cap, dtype=jnp.float64)
    init_occ = jnp.zeros(cap, dtype=bool)
    init_cnt = jnp.array(0, dtype=jnp.int32)

    def _term_body(t: int, state: tuple) -> tuple:
        keys, weights, occ, cnt = state
        cm = create_mask[t]
        am = annihilate_mask[t]
        fm = flip_mask[t]

        def _config_body(i: int, state: tuple) -> tuple:
            keys, weights, occ, cnt = state
            c = configs_i[i]
            applicable = _check_applicable(c, cm, am, n_qubytes)
            new_c = c ^ fm
            excluded = _check_excluded(new_c, configs_exclude)
            processed = applicable & (~excluded)

            def _do_process(_: Array) -> tuple:
                cr = coef[t, 0]
                ci = coef[t, 1]
                pr = psi_i[i, 0]
                pi_v = psi_i[i, 1]
                weight = (cr * pr - ci * pi_v) ** 2 + (cr * pi_v + ci * pr) ** 2

                def _probe_body(s: int, probe_state: tuple) -> tuple:
                    _keys, _weights, _occ, _cnt, _new_c, _w, _matched = probe_state
                    is_match = _occ[s] & jnp.all(_keys[s] == _new_c)

                    def _found() -> tuple:
                        prev = _weights[s]
                        return (
                            _keys,
                            _weights.at[s].set(jnp.maximum(prev, _w)),
                            _occ,
                            _cnt,
                            _new_c,
                            _w,
                            jnp.ones((), dtype=bool),
                        )

                    def _not_found() -> tuple:
                        return (_keys, _weights, _occ, _cnt, _new_c, _w, _matched)

                    return lax.cond(is_match, _found, _not_found)

                new_keys, new_weights, new_occ, new_cnt, _, _, matched = lax.fori_loop(
                    0, cap, _probe_body, (keys, weights, occ, cnt, new_c, weight, jnp.zeros((), dtype=bool))
                )

                def _insert_empty() -> tuple:
                    return (
                        new_keys.at[new_cnt].set(new_c),
                        new_weights.at[new_cnt].set(weight),
                        new_occ.at[new_cnt].set(True),
                        new_cnt + 1,
                    )

                def _keep() -> tuple:
                    return (new_keys, new_weights, new_occ, new_cnt)

                return lax.cond(
                    (~matched) & (new_cnt < cap),
                    _insert_empty,
                    _keep,
                )

            return lax.cond(processed, _do_process, lambda _: (keys, weights, occ, cnt), None)

        return lax.fori_loop(0, batch_size, _config_body, (keys, weights, occ, cnt))

    final_keys, final_weights, _, _ = lax.fori_loop(
        0, term_count, _term_body, (init_keys, init_weights, init_occ, init_cnt)
    )
    idx = jnp.argsort(final_weights)[::-1][:count_selected]
    return final_keys[idx]
