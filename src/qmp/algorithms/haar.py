"""HAAR: Hybrid Adaptive Antisymmetric Representation algorithm.

Two-step optimization:
1. Krylov (Lanczos) imaginary-time evolution with adaptive basis extension
2. Local optimization: gradient descent matching network to Krylov target
"""

from __future__ import annotations

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
            results.append((float(vals[i]), psi))
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
    both_c = jnp.concatenate([ca, cb], axis=0)
    both_p = jnp.concatenate([pa, pb], axis=0)
    flat = both_c.reshape(both_c.shape[0], -1).view(jnp.uint32)
    _, idx, _ = jnp.unique(flat, axis=0, return_index=True, return_counts=True, size=flat.shape[0], fill_value=0)  # ty: ignore — jax size/fill_value
    return both_c[idx], both_p[idx]


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

    def run(self) -> None:
        config = self._config
        if self._model is None or self._network is None:
            logger.error("HAAR requires both model and network to be configured.")
            return

        loss_fn = _AVAILABLE_LOSSES[config.loss_name]

        state: dict[str, typing.Any]
        state = _load_checkpoint(config.checkpoint_path) if config.checkpoint_path else _init_state()
        if state is None:
            state = _init_state()

        cycle = state["haar"]["global"]
        logger.info("HAAR starting from cycle %d", int(cycle))

        while True:
            logger.info("=== Cycle %d ===", int(cycle))
            key = jax.random.key(cycle * config.sampling_count_from_network)

            # --- sample ---
            logger.info("Sampling from network...")
            c_net, p_net, _ = self._network.generate_unique(config.sampling_count_from_network, key=key)  # ty: ignore — network dynamic

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
            max_idx = jnp.argmax(jnp.abs(target_psi))
            target_psi = target_psi / target_psi[max_idx]

            # --- local optimization
            logger.info("Starting local optimization...")
            _new_params, _opt, _step = _local_optimize(
                self._network,
                configs,
                target_psi,
                max_idx,
                loss_fn,
                config.local_max_steps,
                config.local_stop_loss,
            )
            state["haar"]["local"] = _step

            # --- update pool ---
            state["haar"]["pool"] = (configs, psi, jnp.ones_like(psi.real))
            state["haar"]["global"] = cycle + 1
            cycle += 1

            # --- checkpoint ---
            if cycle % config.checkpoint_interval == 0:
                cp_path = config.checkpoint_path or f"haar_checkpoint_cycle{cycle:06d}.pkl"
                _save_checkpoint(state, cp_path)


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
    network: object,
    configs: Array,
    target_psi: Array,
    max_idx: int,
    loss_fn: typing.Callable[..., Array],
    max_steps: int,
    stop_loss: float,
) -> tuple[object, object, int]:
    import copy as _copy

    opt = optax.adam(1e-3)
    opt_state = opt.init(nnx.state(network, nnx.Param))  # ty: ignore — network dynamic

    for try_idx in range(5):
        params_backup = _copy.deepcopy(nnx.state(network, nnx.Param))  # ty: ignore
        opt_backup = _copy.deepcopy(opt_state)

        for step in range(max_steps):

            def _loss_grad(pdict: dict[str, typing.Any]) -> Array:  # ty: ignore — closure
                nnx.update(network, pdict)  # ty: ignore
                psi_net = network(configs)  # ty: ignore — network dynamic
                psi_net = psi_net / psi_net[max_idx]
                return loss_fn(psi_net, target_psi)

            loss_val, grads = jax.value_and_grad(_loss_grad)(nnx.state(network, nnx.Param))  # ty: ignore
            updates, opt_state = opt.update(grads, opt_state, nnx.state(network, nnx.Param))  # ty: ignore
            new_params = optax.apply_updates(nnx.state(network, nnx.Param), updates)  # ty: ignore
            nnx.update(network, new_params)  # ty: ignore

            if step % 100 == 0:
                logger.info("  step %d, loss=%.10f", step, float(loss_val))

            if jnp.isnan(loss_val) or jnp.isinf(loss_val):
                logger.warning("NaN/inf loss at step %d, restoring backup", step)
                nnx.update(network, params_backup)  # ty: ignore
                opt_state = opt_backup
                break

            if float(loss_val) < stop_loss:
                logger.info("Loss threshold met at step %d", step)
                return nnx.state(network, nnx.Param), opt_state, step  # ty: ignore

        else:
            return nnx.state(network, nnx.Param), opt_state, max_steps  # ty: ignore

    logger.error("Local optimization failed after all retries")
    nnx.update(network, params_backup)  # ty: ignore
    return params_backup, opt_backup, 0


# ==============================================================================
# Registration
# ==============================================================================

action_config_dict["haar"] = HaarConfig
action_class_dict["haar"] = Haar
