"""Tests for HAAR config and loss functions."""

from __future__ import annotations

import jax.numpy as jnp

from qmp.algorithms._losses import sum_filtered_angle_scaled_log
from qmp.algorithms.haar import HaarConfig, KrylovBasisStrategy


def test_haar_config_defaults() -> None:
    cfg = HaarConfig()
    assert cfg.basis_strategy == KrylovBasisStrategy.ADAPTIVE
    assert cfg.krylov_max_steps == 32
    assert cfg.krylov_random_period == 31
    assert cfg.local_max_steps == 10000


def test_krylov_strategy_enum_values() -> None:
    assert KrylovBasisStrategy.FIXED.value == "fixed"
    assert KrylovBasisStrategy.ADAPTIVE.value == "adaptive"


def test_loss_identical_gives_zero() -> None:
    psi = jnp.array([1.0 + 0j, 2.0 + 1j])
    loss = sum_filtered_angle_scaled_log(psi, psi)
    assert float(loss) < 1e-10


def test_loss_different_nonzero() -> None:
    a = jnp.array([1.0 + 0j, 0.0 + 0j])
    b = jnp.array([0.0 + 0j, 1.0 + 0j])
    loss = sum_filtered_angle_scaled_log(a, b)
    assert float(loss) > 0


def test_loss_scale_invariant() -> None:
    psi = jnp.array([1.0 + 2j, 0.5 + 1j, 2.0 + 0j])
    loss1 = sum_filtered_angle_scaled_log(psi, psi * 3.0)
    loss2 = sum_filtered_angle_scaled_log(psi, psi * 0.5)
    assert abs(float(loss1 - loss2)) < 1e-10


def test_haar_config_registration() -> None:
    from qmp.algorithms._registry import action_class_dict, action_config_dict
    from qmp.algorithms.haar import Haar

    assert action_config_dict["haar"] is HaarConfig
    assert action_class_dict["haar"] is Haar
