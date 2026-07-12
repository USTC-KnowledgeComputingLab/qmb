"""Shared autoregressive helpers for neural quantum states.

These pure functions are orthogonal to the network backbone (MLP / Transformer)
and to the symmetry variant (Normal / Electron / ElectronUpDown). They implement:

- conditional log-amplitude normalisation
- particle-number conservation masks
- the autoregressive Gumbel top-K trick for sampling without replacement
  (arXiv:2408.07625, Kool et al. JMLR 2020)
- multinomial sampling for sampling with replacement

Log-amplitudes are the natural logarithm of the (unnormalised) amplitude, so the
local probability of a state is ``exp(2 * log_amplitude)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array

# Finite stand-in for -inf, used for invalid or padding beam slots so that the
# Gumbel conditioning arithmetic never evaluates ``inf - inf = NaN``.
INVALID_LOG_PROB = -1e30


def normalize_log_amplitude(log_amplitude: Array, axis: int = -1) -> Array:
    """Normalise conditional log-amplitudes so the local probabilities sum to 1.

    After normalisation ``sum(exp(2 * result)) == 1`` along ``axis``.

    Parameters
    ----------
    log_amplitude : Array
        Unnormalised conditional log-amplitudes.
    axis : int
        Axis over which the local probability distribution is defined.

    Returns
    -------
    Array
        Normalised log-amplitudes with the same shape as the input.
    """
    log_partition = 0.5 * jax.nn.logsumexp(2.0 * log_amplitude, axis=axis, keepdims=True)
    return log_amplitude - log_partition


def apply_mask(log_amplitude: Array, mask: Array) -> Array:
    """Set forbidden states to a log-amplitude of ``-inf`` (zero probability)."""
    return jnp.where(mask, log_amplitude, -jnp.inf)


def _spin_mask(occupied_count: Array, sites_filled: Array, total_sites: int, target_occupied: int) -> Array:
    """Single-species conservation mask over states ``(hole=0, particle=1)``.

    Returns a boolean array of shape ``[..., 2]`` indicating whether a hole or a
    particle may still be appended without breaking the target occupation.
    """
    occupied_count = jnp.asarray(occupied_count)
    sites_filled = jnp.asarray(sites_filled)
    hole_count = sites_filled - occupied_count
    can_add_hole = hole_count < (total_sites - target_occupied)
    can_add_particle = occupied_count < target_occupied
    return jnp.stack([can_add_hole, can_add_particle], axis=-1)


def mask_electron(electron_count: Array, sites_filled: Array, total_sites: int, electrons: int) -> Array:
    """Total electron-number conservation mask.

    Parameters
    ----------
    electron_count : Array
        Number of electrons placed so far (any broadcastable integer shape).
    sites_filled : Array
        Number of sites already decided (broadcastable to ``electron_count``).
    total_sites : int
        Total number of sites.
    electrons : int
        Target total electron number.

    Returns
    -------
    Array
        Boolean array of shape ``[..., 2]`` for states ``(hole, electron)``.
    """
    return _spin_mask(electron_count, sites_filled, total_sites, electrons)


def mask_electron_up_down(
    up_count: Array,
    down_count: Array,
    sites_filled: Array,
    total_sites: int,
    spin_up: int,
    spin_down: int,
) -> Array:
    """Spin-resolved electron-number conservation mask.

    Returns a boolean array of shape ``[..., 2, 2]`` indexed ``[up_state,
    down_state]`` indicating whether the corresponding two-qubit site value may
    be appended without breaking either spin population target.
    """
    up_ok = _spin_mask(up_count, sites_filled, total_sites, spin_up)
    down_ok = _spin_mask(down_count, sites_filled, total_sites, spin_down)
    return up_ok[..., :, None] & down_ok[..., None, :]


def gumbel_topk_step(
    parent_log_prob: Array,
    parent_perturbed: Array,
    parent_valid: Array,
    conditional_log_prob: Array,
    key: Array,
) -> tuple[Array, Array, Array]:
    """One step of the autoregressive Gumbel top-K trick.

    Extends every beam prefix by every possible next state, computing the child
    cumulative log-probabilities and their conditioned Gumbel-perturbed values.

    Parameters
    ----------
    parent_log_prob : Array
        ``[beam_width]`` cumulative unperturbed log-probability of each prefix.
    parent_perturbed : Array
        ``[beam_width]`` conditioned perturbed log-probability of each prefix.
    parent_valid : Array
        ``[beam_width]`` boolean flags marking real (non-padding) prefixes.
    conditional_log_prob : Array
        ``[beam_width, n_states]`` conditional log-probability ``log p(x_i|x_<i)``
        of each next state; forbidden states must be ``-inf``.
    key : Array
        A JAX PRNG key.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(child_log_prob, child_perturbed, child_valid)``, each of shape
        ``[beam_width, n_states]``. Invalid children carry ``INVALID_LOG_PROB``.
    """
    child_valid = jnp.isfinite(conditional_log_prob) & parent_valid[:, None]

    child_log_prob = parent_log_prob[:, None] + conditional_log_prob

    gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(key, conditional_log_prob.shape, dtype=child_log_prob.dtype)))
    perturbed = child_log_prob + gumbel_noise
    perturbed = jnp.where(child_valid, perturbed, INVALID_LOG_PROB)

    child_max = jnp.max(perturbed, axis=-1, keepdims=True)

    # Guard the parent value so padding slots never inject inf into the arithmetic.
    parent_perturbed_safe = jnp.where(parent_valid, parent_perturbed, 0.0)[:, None]
    conditioned = -jnp.log(jnp.exp(-parent_perturbed_safe) - jnp.exp(-child_max) + jnp.exp(-perturbed))

    child_perturbed = jnp.where(child_valid, conditioned, INVALID_LOG_PROB)
    child_log_prob = jnp.where(child_valid, child_log_prob, INVALID_LOG_PROB)
    return child_log_prob, child_perturbed, child_valid


def sample_step(conditional_log_prob: Array, key: Array) -> Array:
    """Sample one next state per batch element from ``exp(2 * conditional_log_prob)``.

    Parameters
    ----------
    conditional_log_prob : Array
        ``[batch_size, n_states]`` conditional log-amplitudes; forbidden states
        must be ``-inf``.
    key : Array
        A JAX PRNG key.

    Returns
    -------
    Array
        ``[batch_size]`` integer indices of the sampled next state.
    """
    logits = 2.0 * conditional_log_prob
    return jax.random.categorical(key, logits, axis=-1)
