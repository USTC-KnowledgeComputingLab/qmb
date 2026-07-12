"""Loss functions for local optimization in HAAR algorithm."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def sum_filtered_angle_scaled_log(psi_net: Array, psi_target: Array) -> Array:
    """Sum of filtered angle-scaled log differences.

    Measures the angle between network and target wavefunctions, weighted
    by target amplitude to focus on dominant configurations.
    """
    target_prob = (psi_target.conj() * psi_target).real
    target_weight = target_prob / (jnp.sum(target_prob) + 1e-30)

    angle_diff = (psi_net.conj() * psi_target).real / (jnp.abs(psi_net) * jnp.abs(psi_target) + 1e-30)
    angle_diff = jnp.clip(angle_diff, -1.0, 1.0)
    angle = jnp.arccos(angle_diff)

    return jnp.sum(target_weight * angle**2)
