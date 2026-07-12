"""TrimCI × HAAR hybrid algorithm.

Combines TrimCI's expansion + two-level (local block / global) trimming
determinant selection with HAAR's neural-quantum-state fitting loop.

One cycle:
1. sample from network + pool
2. multi-hop global top-K expansion (H·psi propagates weights between hops)
3. local trim: random block partition + per-block Lanczos + per-block top-K
4. global trim: global Lanczos, select next core by |psi|
5. build |psi|^2 target
6. fit network to target (gradient descent)
7. update pool + checkpoint
"""

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from qmp.algorithms._registry import action_class_dict, action_config_dict
from qmp.algorithms.haar import (
    _AVAILABLE_LOSSES,
    KrylovBasisStrategy,
    _DynamicLanczos,
    _load_checkpoint,
    _local_optimize,
    _merge_pools,
    _sample_from_pool,
    _save_checkpoint,
)
from qmp.models._build import SubConfigRef, build_model

logger = logging.getLogger(__name__)


# ==============================================================================
# Config
# ==============================================================================


@dataclass
class TrimConfig:
    model: SubConfigRef | None = None
    network: SubConfigRef | None = None

    sampling_count_from_network: int = 1024
    sampling_count_from_pool: int = 1024

    max_rounds: int = 4
    pool_core_ratio: int = 10

    num_groups: int = 10
    local_keep_count: int = 64
    local_lanczos_steps: int = 16

    global_lanczos_steps: int = 32
    krylov_stop_norm: float = 1e-8
    krylov_random_period: int = 31
    krylov_state_count: int = 1
    core_keep_count: int = 128
    first_cycle_keep_size: int = 10

    loss_name: str = "sum_filtered_angle_scaled_log"
    local_max_steps: int = 10000
    local_stop_loss: float = 1e-8

    checkpoint_path: str | None = None
    checkpoint_interval: int = 1
    max_cycles: int = -1


# ==============================================================================
# TrimCI stages
# ==============================================================================


def _unique_configs(configs: Array) -> Array:
    """Deduplicate bit-packed configs, preserving first-occurrence order.

    Mirrors the 4-byte alignment trick in ``haar._merge_pools`` so the uint8
    rows can be viewed as uint32 for ``jnp.unique``.
    """
    n_bytes = configs.shape[1]
    padded = n_bytes if n_bytes % 4 == 0 else ((n_bytes // 4) + 1) * 4
    work = configs
    if n_bytes < padded:
        work = jnp.pad(work, ((0, 0), (0, padded - n_bytes)))
    flat = work.reshape(work.shape[0], -1).view(jnp.uint32)
    _, idx = jnp.unique(flat, axis=0, return_index=True)
    return configs[jnp.sort(idx)]


def _expand_pool(
    model: object,
    core_configs: Array,
    core_psi: Array,
    max_rounds: int,
    pool_core_ratio: int,
) -> tuple[Array, Array]:
    """Multi-hop global top-K expansion.

    Each hop selects the top-K new configs by |H_ij c_j|^2, then propagates
    amplitudes one hop via H·psi to weight the next hop (true graph diffusion).
    The pool is deduplicated each hop (top-K may return zero-padded rows on
    small systems). Returns (pool_configs, pool_psi) where pool_psi is the
    last-hop H·psi projected onto the pool.
    """
    pool_configs, pool_psi = core_configs, core_psi
    for _ in range(max_rounds):
        psi_real = jnp.stack([pool_psi.real, pool_psi.imag], axis=1)
        count = pool_core_ratio * pool_configs.shape[0]
        new_c = model.find_topk_relative_configs(pool_configs, psi_real, count, pool_configs)  # ty: ignore — model dynamic
        if new_c.shape[0] == 0:
            break
        new_pool = _unique_configs(jnp.concatenate([pool_configs, new_c], axis=0))
        hpsi = model.apply_within_subspace(pool_configs, psi_real, new_pool)  # ty: ignore — model dynamic
        pool_configs = new_pool
        pool_psi = hpsi[:, 0] + 1j * hpsi[:, 1]
    return pool_configs, pool_psi


def _local_trim(
    model: object,
    pool_configs: Array,
    pool_psi: Array,
    num_groups: int,
    keep_count: int,
    lanczos_steps: int,
    stop_norm: float,
    random_period: int,
    key: Array,
) -> tuple[Array, Array]:
    """Random block partition + per-block Lanczos + per-block top-K by |c|.

    The pool is randomly permuted and split into ``num_groups`` blocks; each
    block is diagonalized independently (cheap, unbiased, diversity-preserving)
    and its top ``keep_count`` configs by |c| survive. Returns merged,
    deduplicated surviving (configs, psi).
    """
    n = pool_configs.shape[0]
    perm = jax.random.permutation(key, n)
    groups = jnp.array_split(perm, num_groups)

    survived_c: Array | None = None
    survived_p: Array | None = None
    for group_idx in groups:
        if group_idx.shape[0] == 0:
            continue
        block_c = pool_configs[group_idx]
        block_p = pool_psi[group_idx]
        lanczos = _DynamicLanczos(
            model=model,
            configs=block_c,
            psi=block_p,
            max_steps=lanczos_steps,
            stop_norm=stop_norm,
            random_period=random_period,
            extend_count=0,
            strategy=KrylovBasisStrategy.FIXED,
            state_count=1,
        )
        results = list(lanczos.run())
        _e, cfg, ritz = results[-1][0]
        k = min(keep_count, cfg.shape[0])
        top = jnp.argsort((ritz.conj() * ritz).real)[::-1][:k]
        kept_c, kept_p = cfg[top], ritz[top]
        if survived_c is None:
            survived_c, survived_p = kept_c, kept_p
        else:
            survived_c, survived_p = _merge_pools(survived_c, survived_p, kept_c, kept_p)
    assert survived_c is not None and survived_p is not None
    return survived_c, survived_p


def _init_state() -> dict[str, typing.Any]:
    return {
        "trim": {
            "global": 0,
            "local": 0,
            "pool": None,
            "excited": [],
        },
    }


class Trim:
    """TrimCI × HAAR algorithm."""

    def __init__(self, config: TrimConfig) -> None:
        self._config = config
        self._model = build_model(config.model)
        self._network = None
        if config.network is not None and self._model is not None:
            self._network = self._model.create_network(config.network.name, config.network.params, rngs=nnx.Rngs(42))
        self._state: dict[str, typing.Any] = _init_state()

    def run(self) -> None:
        config = self._config
        if self._model is None or self._network is None:
            logger.error("Trim requires both model and network to be configured.")
            return

        loss_fn = _AVAILABLE_LOSSES[config.loss_name]

        state: dict[str, typing.Any]
        state = _load_checkpoint(config.checkpoint_path) if config.checkpoint_path else _init_state()
        if state is None:
            state = _init_state()
        self._state = state

        cycle = state["trim"]["global"]
        start = cycle
        logger.info("Trim starting from cycle %d", int(cycle))

        while config.max_cycles < 0 or cycle < start + config.max_cycles:
            logger.info("=== Cycle %d ===", int(cycle))
            key = jax.random.key(cycle * config.sampling_count_from_network)

            # --- 1. sample ---
            c_net, p_net = self._network.generate_unique(config.sampling_count_from_network, key=key)  # ty: ignore — network dynamic
            key2 = jax.random.fold_in(key, 1)
            c_pool, p_pool = _sample_from_pool(state["trim"]["pool"], config.sampling_count_from_pool, key2)
            core_c, core_p = _merge_pools(c_net, p_net, c_pool, p_pool)
            logger.info("Core: %d unique configs", int(core_c.shape[0]))

            # --- 2. expansion ---
            pool_c, pool_p = _expand_pool(self._model, core_c, core_p, config.max_rounds, config.pool_core_ratio)
            logger.info("Pool after expansion: %d configs", int(pool_c.shape[0]))

            # --- 3. local trim ---
            key3 = jax.random.fold_in(key, 2)
            surv_c, surv_p = _local_trim(
                self._model,
                pool_c,
                pool_p,
                config.num_groups,
                config.local_keep_count,
                config.local_lanczos_steps,
                config.krylov_stop_norm,
                config.krylov_random_period,
                key3,
            )
            logger.info("Survived local trim: %d configs", int(surv_c.shape[0]))

            # --- 4. global trim ---
            lanczos = _DynamicLanczos(
                model=self._model,
                configs=surv_c,
                psi=surv_p,
                max_steps=config.global_lanczos_steps,
                stop_norm=config.krylov_stop_norm,
                random_period=config.krylov_random_period,
                extend_count=0,
                strategy=KrylovBasisStrategy.FIXED,
                state_count=config.krylov_state_count,
            )
            results: list[tuple[float, Array, Array]] = []
            for results in lanczos.run():
                e0, _cfg, _psi0 = results[0]
                logger.info("Global Krylov energy: %.10f (basis %d)", e0, int(_cfg.shape[0]))
            _e0, configs, psi = results[0]
            state["trim"]["excited"] = [(e, cfg, p) for e, cfg, p in results]

            # --- 5. target construction ---
            target_prob = jnp.zeros_like(psi, dtype=jnp.float64)
            for _e, _cfg, p in results:
                target_prob = target_prob + (p.conj() * p).real
            target_psi = jnp.sqrt(target_prob).astype(jnp.complex128)
            max_idx = jnp.argmax(jnp.abs(target_psi))
            target_psi = target_psi / target_psi[max_idx]

            # --- 6. local optimization ---
            logger.info("Starting local optimization...")
            _new_params, _opt, step = _local_optimize(
                self._network,
                configs,
                target_psi,
                max_idx,
                loss_fn,
                config.local_max_steps,
                config.local_stop_loss,
            )
            state["trim"]["local"] = step

            # --- 7. update pool (next core = top-keep by |psi|) + checkpoint ---
            keep = config.first_cycle_keep_size if cycle == 0 else config.core_keep_count
            keep = min(keep, configs.shape[0])
            order = jnp.argsort((psi.conj() * psi).real)[::-1][:keep]
            core_sel_c, core_sel_p = configs[order], psi[order]
            state["trim"]["pool"] = (core_sel_c, core_sel_p, jnp.ones_like(core_sel_p.real))
            state["trim"]["global"] = cycle + 1
            cycle += 1

            if cycle % config.checkpoint_interval == 0 and config.checkpoint_path is not None:
                _save_checkpoint(state, config.checkpoint_path)


# ==============================================================================
# Registration
# ==============================================================================

action_config_dict["trim"] = TrimConfig
action_class_dict["trim"] = Trim
