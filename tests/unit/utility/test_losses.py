"""Tests for loss functions."""

from __future__ import annotations

import math

import jax.numpy as jnp

from qmp.utility._losses import (
    _scaled_abs,
    _scaled_angle,
    direct,
    log,
    sum_filtered_angle_scaled_log,
)

# ---- helpers ----


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


# ---- log ----


def test_log_identical_gives_zero() -> None:
    psi = jnp.array([1.0 + 0j, 2.0 + 1j])
    assert float(log(psi, psi)) < 1e-10


# ---- sum_filtered_angle_scaled_log ----


def test_sum_filtered_angle_scaled_log_identical_gives_zero() -> None:
    psi = jnp.array([1.0 + 0j, 2.0 + 1j])
    loss = sum_filtered_angle_scaled_log(psi, psi)
    assert float(loss) < 1e-10


def test_sum_filtered_angle_scaled_log_different_nonzero() -> None:
    a = jnp.array([1.0 + 0j, 0.0 + 0j])
    b = jnp.array([0.0 + 0j, 1.0 + 0j])
    loss = sum_filtered_angle_scaled_log(a, b)
    assert float(loss) > 0


# ---- direct ----


def test_direct_identical_gives_zero() -> None:
    psi = jnp.array([1.0 + 0j, 2.0 + 1j])
    assert float(direct(psi, psi)) < 1e-10


def test_direct_different_nonzero() -> None:
    a = jnp.array([1.0 + 0j, 0.0 + 0j])
    b = jnp.array([0.0 + 0j, 1.0 + 0j])
    assert float(direct(a, b)) > 0
