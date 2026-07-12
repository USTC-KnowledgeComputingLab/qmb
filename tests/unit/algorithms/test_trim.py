"""Tests for TrimCI config, helpers, and registration."""

from __future__ import annotations

from qmp.algorithms.trim import Trim, TrimConfig


def test_trim_config_defaults() -> None:
    cfg = TrimConfig()
    assert cfg.max_rounds == 4
    assert cfg.pool_core_ratio == 10
    assert cfg.num_groups == 10
    assert cfg.local_keep_count == 64
    assert cfg.core_keep_count == 128
    assert cfg.first_cycle_keep_size == 10
    assert cfg.max_cycles == -1
    assert cfg.loss_name == "sum_filtered_angle_scaled_log"


def test_trim_registration() -> None:
    from qmp.algorithms._registry import action_class_dict, action_config_dict

    assert action_config_dict["trim"] is TrimConfig
    assert action_class_dict["trim"] is Trim


import jax
import jax.numpy as jnp
from flax import nnx

from qmp.algorithms.trim import _expand_pool
from qmp.models.hubbard import Model, ModelConfig


def _small_hubbard() -> Model:
    return Model(ModelConfig(m=2, n=1, t=1.0, u=4.0, electron_number=2))


def test_expand_pool_grows_and_unique() -> None:
    model = _small_hubbard()
    net = model.create_network("mlp/u1u1", {"hidden_size": [8]}, rngs=nnx.Rngs(0))
    core_c, core_p = net.generate_unique(4, key=jax.random.key(1))
    pool_c, pool_p = _expand_pool(model, core_c, core_p, max_rounds=2, pool_core_ratio=3)

    assert pool_c.shape[0] >= core_c.shape[0]
    assert pool_c.shape[0] == pool_p.shape[0]
    flat = pool_c.reshape(pool_c.shape[0], -1)
    unique = jnp.unique(flat, axis=0)
    assert unique.shape[0] == pool_c.shape[0]  # no duplicate configs


from qmp.algorithms.trim import _local_trim


def test_local_trim_reduces_and_unique() -> None:
    model = _small_hubbard()
    net = model.create_network("mlp/u1u1", {"hidden_size": [8]}, rngs=nnx.Rngs(0))
    core_c, core_p = net.generate_unique(4, key=jax.random.key(2))
    pool_c, pool_p = _expand_pool(model, core_c, core_p, max_rounds=2, pool_core_ratio=3)

    surv_c, surv_p = _local_trim(
        model,
        pool_c,
        pool_p,
        num_groups=2,
        keep_count=2,
        lanczos_steps=4,
        stop_norm=1e-8,
        random_period=99,
        key=jax.random.key(3),
    )

    assert surv_c.shape[0] == surv_p.shape[0]
    assert surv_c.shape[0] <= pool_c.shape[0]
    assert surv_c.shape[0] <= 2 * 2  # num_groups * keep_count upper bound
    flat = surv_c.reshape(surv_c.shape[0], -1)
    assert jnp.unique(flat, axis=0).shape[0] == surv_c.shape[0]
