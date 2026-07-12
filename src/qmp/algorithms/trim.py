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

    def run(self) -> None:
        raise NotImplementedError


# ==============================================================================
# Registration
# ==============================================================================

action_config_dict["trim"] = TrimConfig
action_class_dict["trim"] = Trim
