"""Loss functions for local optimization.

These compute the difference between network wavefunction (s) and
Krylov target wavefunction (t).  All functions work on complex128 arrays
of shape ``[batch]`` and return a scalar float64 loss.

Catalogue
---------
log                              — basic log-amplitude + phase difference
sum_reweighted_log               — reweighted by total amplitude
sum_filtered_log                 — filtered by scaled angle at low amplitude
sum_filtered_scaled_log          — filtered + scaled-abs for tiny amplitudes
sum_reweighted_angle_log         — only phase reweighted
sum_filtered_angle_log           — only phase filtered
sum_filtered_angle_scaled_log    — scaled-abs + only-phase filtered
direct                           — plain |s-t|²

Helper semantics (ported from old torch version)
------------------------------------------------
_scaled_abs(x, m) : log(x) for x > m, else (x-m)/m + log(m)
                     Avoids log(0) explosion.
_scaled_angle(x, m) : 1 / (1 + m/x)
                       → 0 when x ≪ m, → 1 when x ≫ m.
                       Used to suppress phase gradients for low-amplitude
                       configurations where the phase is ill-defined.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scaled_abs(x: Array, m: float) -> Array:
    """Log-like function linearised below *m* to avoid blow-up."""
    return jnp.where(x > m, jnp.log(x), (x - m) / m + math.log(m))


def _scaled_angle(scale: Array, m: float) -> Array:
    """Weight factor → 0 for small amplitudes, → 1 for large."""
    return 1.0 / (1.0 + m / scale)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def log(s: Array, t: Array) -> Array:
    s_abs = jnp.abs(s)
    t_abs = jnp.abs(t)
    s_angle = jnp.angle(s)
    t_angle = jnp.angle(t)
    s_mag = jnp.log(s_abs)
    t_mag = jnp.log(t_abs)
    e_real = (s_mag - t_mag) / (2.0 * math.pi)
    e_imag = (s_angle - t_angle) / (2.0 * math.pi)
    e_imag = e_imag - jnp.round(e_imag)
    return jnp.mean(e_real**2 + e_imag**2)


def sum_reweighted_log(s: Array, t: Array) -> Array:
    s_abs = jnp.abs(s)
    t_abs = jnp.abs(t)
    s_angle = jnp.angle(s)
    t_angle = jnp.angle(t)
    s_mag = jnp.log(s_abs)
    t_mag = jnp.log(t_abs)
    e_real = (s_mag - t_mag) / (2.0 * math.pi)
    e_imag = (s_angle - t_angle) / (2.0 * math.pi)
    e_imag = e_imag - jnp.round(e_imag)
    loss = e_real**2 + e_imag**2
    loss = loss * (t_abs + s_abs)
    return jnp.mean(loss)


def sum_filtered_log(s: Array, t: Array, min_magnitude: float = 1e-10) -> Array:
    s_abs = jnp.abs(s)
    t_abs = jnp.abs(t)
    s_angle = jnp.angle(s)
    t_angle = jnp.angle(t)
    s_mag = jnp.log(s_abs)
    t_mag = jnp.log(t_abs)
    e_real = (s_mag - t_mag) / (2.0 * math.pi)
    e_imag = (s_angle - t_angle) / (2.0 * math.pi)
    e_imag = e_imag - jnp.round(e_imag)
    loss = e_real**2 + e_imag**2
    loss = loss * _scaled_angle(t_abs + s_abs, min_magnitude)
    return jnp.mean(loss)


def sum_filtered_scaled_log(s: Array, t: Array, min_magnitude: float = 1e-10) -> Array:
    s_abs = jnp.abs(s)
    t_abs = jnp.abs(t)
    s_angle = jnp.angle(s)
    t_angle = jnp.angle(t)
    s_mag = _scaled_abs(s_abs, min_magnitude)
    t_mag = _scaled_abs(t_abs, min_magnitude)
    e_real = (s_mag - t_mag) / (2.0 * math.pi)
    e_imag = (s_angle - t_angle) / (2.0 * math.pi)
    e_imag = e_imag - jnp.round(e_imag)
    loss = e_real**2 + e_imag**2
    loss = loss * _scaled_angle(t_abs + s_abs, min_magnitude)
    return jnp.mean(loss)


def sum_reweighted_angle_log(s: Array, t: Array) -> Array:
    s_abs = jnp.abs(s)
    t_abs = jnp.abs(t)
    s_angle = jnp.angle(s)
    t_angle = jnp.angle(t)
    s_mag = jnp.log(s_abs)
    t_mag = jnp.log(t_abs)
    e_real = (s_mag - t_mag) / (2.0 * math.pi)
    e_imag = (s_angle - t_angle) / (2.0 * math.pi)
    e_imag = e_imag - jnp.round(e_imag)
    e_imag = e_imag * (t_abs + s_abs)
    return jnp.mean(e_real**2 + e_imag**2)


def sum_filtered_angle_log(s: Array, t: Array, min_magnitude: float = 1e-10) -> Array:
    s_abs = jnp.abs(s)
    t_abs = jnp.abs(t)
    s_angle = jnp.angle(s)
    t_angle = jnp.angle(t)
    s_mag = jnp.log(s_abs)
    t_mag = jnp.log(t_abs)
    e_real = (s_mag - t_mag) / (2.0 * math.pi)
    e_imag = (s_angle - t_angle) / (2.0 * math.pi)
    e_imag = e_imag - jnp.round(e_imag)
    e_imag = e_imag * _scaled_angle(t_abs + s_abs, min_magnitude)
    return jnp.mean(e_real**2 + e_imag**2)


def sum_filtered_angle_scaled_log(s: Array, t: Array, min_magnitude: float = 1e-10) -> Array:
    s_abs = jnp.abs(s)
    t_abs = jnp.abs(t)
    s_angle = jnp.angle(s)
    t_angle = jnp.angle(t)
    s_mag = _scaled_abs(s_abs, min_magnitude)
    t_mag = _scaled_abs(t_abs, min_magnitude)
    e_real = (s_mag - t_mag) / (2.0 * math.pi)
    e_imag = (s_angle - t_angle) / (2.0 * math.pi)
    e_imag = e_imag - jnp.round(e_imag)
    e_imag = e_imag * _scaled_angle(t_abs + s_abs, min_magnitude)
    return jnp.mean(e_real**2 + e_imag**2)


def direct(s: Array, t: Array) -> Array:
    error = s - t
    return jnp.mean(error.real**2 + error.imag**2)
