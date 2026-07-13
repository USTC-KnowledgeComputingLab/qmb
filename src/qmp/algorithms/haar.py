"""HAAR: Hybrid Adaptive Antisymmetric Representation algorithm.

Two-step optimization:
1. Krylov (Lanczos) imaginary-time evolution with adaptive basis extension
2. Local optimization: gradient descent matching network to Krylov target
"""

from __future__ import annotations

import copy as _copy
import dataclasses
import logging
import pickle
import typing
from dataclasses import dataclass, field
from enum import Enum

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array

from qmp.algorithms._registry import action_class_dict, action_config_dict
from qmp.models._build import SubConfigRef, build_model
from qmp.utility._losses import (
    direct,
    log,
    sum_filtered_angle_log,
    sum_filtered_angle_scaled_log,
    sum_filtered_log,
    sum_filtered_scaled_log,
    sum_reweighted_angle_log,
    sum_reweighted_log,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Config
# ==============================================================================


class KrylovBasisStrategy(Enum):
    FIXED = "fixed"
    PRECOMPUTE = "precompute"
    POSTCOMPUTE = "postcompute"
    ADAPTIVE = "adaptive"


@dataclass
class HaarConfig:
    model: SubConfigRef | None = None
    network: SubConfigRef | None = None

    sampling_count_from_network: int = 1024
    sampling_count_from_pool: int = 1024

    krylov_max_steps: int = 32
    krylov_stop_norm: float = 1e-8
    krylov_random_period: int = 31
    krylov_state_count: int = 1
    basis_extend_count: int = 64
    basis_strategy: KrylovBasisStrategy = field(default=KrylovBasisStrategy.ADAPTIVE)

    loss_name: str = "sum_filtered_angle_scaled_log"
    local_max_steps: int = 10000
    local_stop_loss: float = 1e-8
    local_log_psi_count: int = 30

    checkpoint_path: str | None = None
    checkpoint_interval: int = 1
    max_cycles: int = -1  # -1 = infinite loop, >0 = run exactly N cycles then return


# ==============================================================================
# Helpers
# ==============================================================================


def _apply_hamiltonian(model: object, configs: Array, psi: Array) -> Array:
    psi_real = jnp.stack([psi.real, psi.imag], axis=1)
    h_psi = model.apply_within_subspace(configs, psi_real, configs)  # ty: ignore — model is dynamic
    return h_psi[:, 0] + 1j * h_psi[:, 1]


# ==============================================================================
# Dynamic Lanczos
# ==============================================================================


@dataclasses.dataclass
class _DynamicLanczos:
    model: object
    configs: Array
    psi: Array
    max_steps: int
    stop_norm: float
    random_period: int
    extend_count: int
    strategy: KrylovBasisStrategy
    state_count: int

    def _extend(self, psi_weight: Array, basic_configs: Array | None = None) -> None:
        if basic_configs is None:
            basic_configs = self.configs
        n_core = self.configs.shape[0]
        new_c = self.model.find_topk_relative_configs(  # ty: ignore — model is dynamic
            basic_configs, psi_weight, self.extend_count, self.configs
        )
        self.configs = jnp.concatenate([self.configs, new_c], axis=0)
        n_selected = self.configs.shape[0]
        self.psi = jnp.pad(self.psi, (0, n_selected - n_core))
        logger.info("Basis extended from %d to %d", int(n_core), int(n_selected))

    def _run(self) -> typing.Iterator[tuple[list[Array], list[Array], list[Array]]]:
        v: list[Array] = [self.psi / jnp.linalg.norm(self.psi)]
        alpha: list[Array] = []
        beta: list[Array] = []

        w = _apply_hamiltonian(self.model, self.configs, v[-1])
        alpha.append((w.conj() @ v[-1]).real)
        yield (alpha, beta, v)
        w = w - alpha[-1] * v[-1]

        while len(v) <= self.max_steps:
            norm_w = jnp.linalg.norm(w)
            should_inject = (norm_w < self.stop_norm) or (
                self.random_period < self.max_steps and len(v) % self.random_period == 0
            )

            if should_inject:
                key = jax.random.key(len(v))
                random_v = jax.random.normal(key, v[-1].shape, dtype=v[-1].dtype)
                nonzero = jnp.zeros_like(random_v, dtype=bool)
                for prev_v in v:
                    nonzero = nonzero | (jnp.abs(prev_v) > self.stop_norm)
                random_v = jnp.where(nonzero, random_v, 0)
                for prev_v in v:
                    dot = prev_v.conj() @ random_v
                    random_v = random_v - dot * prev_v
                random_v = random_v / jnp.linalg.norm(random_v)
                v.append(random_v)
                beta.append(jnp.array(0.0))
            else:
                beta.append(norm_w)
                v.append(w / norm_w)

            w = _apply_hamiltonian(self.model, self.configs, v[-1])
            alpha.append((w.conj() @ v[-1]).real)
            yield (alpha, beta, v)
            w = w - alpha[-1] * v[-1]
            if len(beta) > 0:
                w = w - beta[-1] * v[-2]

            for prev_v in v:
                w = w - (prev_v.conj() @ w) * prev_v

    def _eigh(self, alpha: list[Array], beta: list[Array], v: list[Array]) -> list[tuple[float, Array]]:
        if len(beta) == 0:
            return [(float(alpha[0]), v[0])]
        n = len(alpha)
        h_tri = jnp.zeros((n, n), dtype=jnp.float64)
        h_tri = h_tri.at[jnp.arange(n), jnp.arange(n)].set(jnp.array([float(a) for a in alpha]))
        h_tri = h_tri.at[jnp.arange(n - 1), jnp.arange(1, n)].set(jnp.array([float(b) for b in beta]))
        h_tri = h_tri.at[jnp.arange(1, n), jnp.arange(n - 1)].set(jnp.array([float(b) for b in beta]))
        vals, vecs = jnp.linalg.eigh(h_tri)
        k = min(self.state_count, n)
        results: list[tuple[float, Array]] = []
        for i in range(k):
            psi = sum(float(vecs[j, i]) * v[j] for j in range(n))
            results.append((float(vals[i]), psi))  # ty: ignore — dynamic types
        return results

    def run(self) -> typing.Iterator[list[tuple[float, Array, Array]]]:
        def _package(results: list[tuple[float, Array]]) -> list[tuple[float, Array, Array]]:
            return [(e, self.configs, p) for e, p in results]

        if self.extend_count == 0 or self.strategy == KrylovBasisStrategy.FIXED:
            for __, (a, b, v_current) in zip(range(1 + self.max_steps), self._run(), strict=False):
                yield _package(self._eigh(a, b, v_current))

        elif self.strategy == KrylovBasisStrategy.PRECOMPUTE:
            psi_current = self.psi
            for __ in range(self.max_steps):
                selected = jnp.argsort((psi_current.conj() * psi_current).real)[-self.extend_count :]
                cfg = self.configs
                self._extend(psi_current[selected], self.configs[selected])
                psi_current = _apply_hamiltonian(self.model, cfg, psi_current)
            for __, (a, b, v_current) in zip(range(1 + self.max_steps), self._run(), strict=False):
                yield _package(self._eigh(a, b, v_current))

        elif self.strategy == KrylovBasisStrategy.POSTCOMPUTE:
            saved: list[tuple[list[Array], list[Array], list[Array]]] = []
            for __, (a, b, v_current) in zip(range(1 + self.max_steps), self._run(), strict=False):
                saved.append((a, b, v_current))
                yield _package(self._eigh(a, b, v_current))
            v_sum = sum((vi.conj() * vi).real for (_, _, v_list) in saved for vi in v_list)
            v_sum = jnp.sqrt(v_sum)
            self._extend(v_sum)
            for __, (a, b, v_current) in zip(range(1 + self.max_steps), self._run(), strict=False):
                yield _package(self._eigh(a, b, v_current))

        else:  # ADAPTIVE
            last_r: tuple[list[Array], list[Array], list[Array]] | None = None
            for step in range(1 + self.max_steps):
                for __, (a, b, v_current) in zip(range(1 + step), self._run(), strict=False):
                    last_r = (a, b, v_current)
                assert last_r is not None
                yield _package(self._eigh(last_r[0], last_r[1], last_r[2]))
                if step != self.max_steps:
                    self._extend(last_r[2][-1])


# ==============================================================================
# Sampling helpers
# ==============================================================================


def _sample_from_pool(pool: tuple | None, count: int, key: Array) -> tuple[Array, Array]:
    if pool is None:
        return (
            jnp.zeros((0, 1), dtype=jnp.uint8),
            jnp.zeros((0,), dtype=jnp.complex128),
        )
    configs, psi, _counts = pool
    prob = (psi.conj() * psi).real
    prob = prob / (jnp.sum(prob) + 1e-30)
    log_prob = jnp.log(prob + 1e-30)
    g = log_prob + jax.random.gumbel(key, log_prob.shape)
    n = min(count, len(g))
    _, indices = jax.lax.top_k(g, n)
    return configs[indices], psi[indices]


def _merge_pools(ca: Array, pa: Array, cb: Array, pb: Array) -> tuple[Array, Array]:
    if cb.shape[0] == 0:
        return ca, pa
    # pool first so that for duplicate configs pool psi wins (return_index picks first)
    both_c = jnp.concatenate([cb, ca], axis=0)
    both_p = jnp.concatenate([pb, pa], axis=0)
    n_bytes = both_c.shape[1]
    padded = n_bytes if n_bytes % 4 == 0 else ((n_bytes // 4) + 1) * 4
    if n_bytes < padded:
        both_c = jnp.pad(both_c, ((0, 0), (0, padded - n_bytes)))
    flat = both_c.reshape(both_c.shape[0], -1).view(jnp.uint32)
    _, idx = jnp.unique(flat, axis=0, return_index=True)
    return both_c[idx, :n_bytes], both_p[idx]


# ==============================================================================
# Checkpoint  (pickle top-level imports at start of file)
# ==============================================================================


def _save_checkpoint(state: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(state, f)
    logger.info("Checkpoint saved to %s", path)


def _load_checkpoint(path: str) -> dict | None:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


# ==============================================================================
# HAAR main
# ==============================================================================

_AVAILABLE_LOSSES: dict[str, typing.Callable[..., Array]] = {
    "log": log,
    "sum_reweighted_log": sum_reweighted_log,
    "sum_filtered_log": sum_filtered_log,
    "sum_filtered_scaled_log": sum_filtered_scaled_log,
    "sum_reweighted_angle_log": sum_reweighted_angle_log,
    "sum_filtered_angle_log": sum_filtered_angle_log,
    "sum_filtered_angle_scaled_log": sum_filtered_angle_scaled_log,
    "direct": direct,
}


class Haar:
    """HAAR algorithm: Krylov subspace + local optimization."""

    def __init__(self, config: HaarConfig) -> None:
        self._config = config
        self._model = build_model(config.model)
        self._network = None
        if config.network is not None and self._model is not None:
            self._network = self._model.create_network(config.network.name, config.network.params, rngs=nnx.Rngs(42))
        self._state: dict[str, typing.Any] = _init_state()

    def run(self) -> None:
        config = self._config
        if self._model is None or self._network is None:
            logger.error("HAAR requires both model and network to be configured.")
            return

        loss_fn = _AVAILABLE_LOSSES[config.loss_name]

        state: dict[str, typing.Any]
        loaded = _load_checkpoint(config.checkpoint_path) if config.checkpoint_path else None
        state = loaded if loaded is not None else _init_state()
        self._state = state

        cycle = state["haar"]["global"]
        start = cycle
        logger.info("HAAR starting from cycle %d", int(cycle))

        while config.max_cycles < 0 or cycle < start + config.max_cycles:
            logger.info("=== Cycle %d ===", int(cycle))
            key = jax.random.key(cycle * config.sampling_count_from_network)

            # --- sample ---
            logger.info("Sampling from network...")
            c_net, p_net = self._network.generate_unique(config.sampling_count_from_network, key=key)

            logger.info("Sampling from pool...")
            key2 = jax.random.fold_in(key, 1)
            c_pool, p_pool = _sample_from_pool(state["haar"]["pool"], config.sampling_count_from_pool, key2)

            configs, psi = _merge_pools(c_net, p_net, c_pool, p_pool)
            logger.info("Merged: %d unique configs", int(configs.shape[0]))

            # --- Krylov ---
            logger.info("Starting Krylov (strategy=%s)...", config.basis_strategy.value)
            lanczos = _DynamicLanczos(
                model=self._model,
                configs=configs,
                psi=psi,
                max_steps=config.krylov_max_steps,
                stop_norm=config.krylov_stop_norm,
                random_period=config.krylov_random_period,
                extend_count=config.basis_extend_count,
                strategy=config.basis_strategy,
                state_count=config.krylov_state_count,
            )

            results: list[tuple[float, Array, Array]] = []
            for results in lanczos.run():
                e0, _cfg, _psi0 = results[0]
                logger.info("Krylov energy: %.10f (basis size %d)", e0, int(_cfg.shape[0]))

            _e0, configs, psi = results[0]
            state["haar"]["excited"] = [(e, cfg, p) for e, cfg, p in results]

            # --- target construction ---
            target_prob = jnp.zeros_like(psi, dtype=jnp.float64)
            for _e, _cfg, p in results:
                target_prob = target_prob + (p.conj() * p).real
            target_psi = jnp.sqrt(target_prob).astype(jnp.complex128)
            max_idx = int(jnp.argmax(jnp.abs(target_psi)))
            target_psi = target_psi / target_psi[max_idx]

            # --- local optimization
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
            state["haar"]["local"] = step

            # --- update pool ---
            state["haar"]["pool"] = (configs, psi, jnp.ones_like(psi.real))
            state["haar"]["global"] = cycle + 1
            cycle += 1

            # --- checkpoint ---
            if cycle % config.checkpoint_interval == 0 and config.checkpoint_path is not None:
                _save_checkpoint(state, config.checkpoint_path)


def _init_state() -> dict[str, typing.Any]:
    return {
        "haar": {
            "global": 0,
            "local": 0,
            "pool": None,
            "excited": [],
        },
    }


def _local_optimize(
    network: typing.Any,
    configs: typing.Any,
    target_psi: typing.Any,
    max_idx: typing.Any,
    loss_fn: typing.Any,
    max_steps: typing.Any,
    stop_loss: typing.Any,
) -> typing.Any:

    graphdef, params = nnx.split(network, nnx.Param)

    def _loss_grad(pdict: dict[str, typing.Any]) -> Array:
        net = nnx.merge(graphdef, pdict)
        psi_net = net(configs)
        psi_net = psi_net / psi_net[max_idx]
        return loss_fn(psi_net, target_psi)

    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    for _ in range(5):
        params_backup = _copy.deepcopy(params)
        opt_backup = _copy.deepcopy(opt_state)
        last_loss: float = 0.0

        success = True
        for step in range(max_steps):
            loss_val, grads = jax.value_and_grad(_loss_grad)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            if step % 100 == 0:
                logger.info("  step %d, loss=%.10f", step, float(loss_val))

            if jnp.isnan(loss_val) or jnp.isinf(loss_val):
                logger.warning("NaN/inf loss at step %d, restoring backup", step)
                success = False
                break

            if float(loss_val) < stop_loss:
                logger.info("Loss threshold met at step %d", step)
                nnx.update(network, params)
                return params, opt_state, step

            if step > 0 and abs(float(loss_val) - last_loss) < stop_loss:
                logger.info("Loss stagnated at step %d", step)
                nnx.update(network, params)
                return params, opt_state, step

            last_loss = float(loss_val)

        else:
            if success:
                # check for NaN/inf in all parameters after optimization
                params_ok = all(
                    not (bool(jnp.any(jnp.isnan(v))) or bool(jnp.any(jnp.isinf(v))))
                    for v in jax.tree_util.tree_leaves(params)
                )
                if not params_ok:
                    logger.warning("NaN detected in parameters, restoring backup")
                    params = params_backup
                    opt_state = opt_backup
                    continue
                nnx.update(network, params)
                return params, opt_state, max_steps

        params = params_backup
        opt_state = opt_backup

    logger.error("Local optimization failed after all retries")
    nnx.update(network, params_backup)
    return params_backup, opt_state, 0


# ==============================================================================
# Registration
# ==============================================================================

action_config_dict["haar"] = HaarConfig
action_class_dict["haar"] = Haar
