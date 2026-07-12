"""Tests for HAAR config and loss functions."""

from __future__ import annotations

import math

import jax.numpy as jnp

from qmp.utility._losses import (
    log,
    direct,
    sum_filtered_angle_scaled_log,
    _scaled_abs,
    _scaled_angle,
)
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


def test_haar_config_registration() -> None:
    from qmp.algorithms._registry import action_class_dict, action_config_dict
    from qmp.algorithms.haar import Haar

    assert action_config_dict["haar"] is HaarConfig
    assert action_class_dict["haar"] is Haar


# ---- loss helpers ----


def test_scaled_abs_large_values() -> None:
    x = jnp.array([2.0, 10.0])
    result = _scaled_abs(x, 0.1)
    expected = jnp.log(x)
    assert jnp.allclose(result, expected, rtol=1e-10)


def test_scaled_abs_small_values() -> None:
    m = 0.1
    x = jnp.array([0.01])
    result = _scaled_abs(x, m)
    expected = (x - m) / m + math.log(m)
    assert jnp.allclose(result, expected, rtol=1e-10)


def test_scaled_angle_weight_range() -> None:
    w = _scaled_angle(jnp.array([100.0]), 1e-10)
    assert float(w[0]) > 0.9
    w2 = _scaled_angle(jnp.array([1e-12]), 1e-10)
    assert float(w2[0]) < 0.1


# ---- other loss functions ----


def test_log_identical_gives_zero() -> None:
    psi = jnp.array([1.0 + 0j, 2.0 + 1j])
    assert float(log(psi, psi)) < 1e-10


def test_direct_identical_gives_zero() -> None:
    psi = jnp.array([1.0 + 0j, 2.0 + 1j])
    assert float(direct(psi, psi)) < 1e-10


def test_direct_different_nonzero() -> None:
    a = jnp.array([1.0 + 0j, 0.0 + 0j])
    b = jnp.array([0.0 + 0j, 1.0 + 0j])
    assert float(direct(a, b)) > 0
