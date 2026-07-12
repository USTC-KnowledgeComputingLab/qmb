"""Tests for shared autoregressive helpers.

Covers: conditional normalisation, particle-number masks, the Gumbel top-K
step (numerics and no-NaN guards), and multinomial sampling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qmp.networks._autoregressive import (
    INVALID_LOG_PROB,
    apply_mask,
    gumbel_topk_step,
    mask_electron,
    mask_electron_up_down,
    normalize_log_amplitude,
    sample_step,
)

# ---- normalisation ----


def test_normalize_probability_sums_to_one() -> None:
    log_amplitude = jnp.array([[0.3, -1.2, 0.7, 2.1]])
    normalized = normalize_log_amplitude(log_amplitude, axis=-1)
    probability = jnp.sum(jnp.exp(2.0 * normalized), axis=-1)
    assert jnp.allclose(probability, 1.0)


def test_normalize_handles_masked_states() -> None:
    log_amplitude = apply_mask(jnp.array([[0.5, 0.5]]), jnp.array([[True, False]]))
    normalized = normalize_log_amplitude(log_amplitude, axis=-1)
    probability = jnp.exp(2.0 * normalized)
    assert jnp.allclose(probability[0, 0], 1.0)
    assert jnp.allclose(probability[0, 1], 0.0)


# ---- masks ----


def test_mask_electron_forbids_excess_electrons() -> None:
    # 2 sites filled, both electrons, target electrons = 2 -> cannot add another electron.
    mask = mask_electron(electron_count=jnp.array(2), sites_filled=jnp.array(2), total_sites=4, electrons=2)
    assert bool(mask[0])  # hole allowed
    assert not bool(mask[1])  # electron forbidden


def test_mask_electron_forbids_excess_holes() -> None:
    # total_sites=4, electrons=2 -> at most 2 holes. Already 2 holes -> must add electron.
    mask = mask_electron(electron_count=jnp.array(0), sites_filled=jnp.array(2), total_sites=4, electrons=2)
    assert not bool(mask[0])  # hole forbidden
    assert bool(mask[1])  # electron allowed


def test_mask_electron_up_down_shape_and_logic() -> None:
    mask = mask_electron_up_down(
        up_count=jnp.array(1),
        down_count=jnp.array(0),
        sites_filled=jnp.array(1),
        total_sites=2,
        spin_up=1,
        spin_down=1,
    )
    assert mask.shape == (2, 2)
    # up already saturated (1 of 1) -> cannot add up electron.
    # up_state index 1 means "add up electron" -> forbidden for both down states.
    assert not bool(mask[1, 0])
    assert not bool(mask[1, 1])
    # down already used its only allowed hole (1 of total-spin_down=1) -> must add down electron.
    assert not bool(mask[0, 0])  # up hole + down hole forbidden
    assert bool(mask[0, 1])  # up hole + down electron allowed


def test_mask_electron_up_down_all_allowed_when_empty() -> None:
    # Large lattice, nothing placed yet -> all four combinations allowed.
    mask = mask_electron_up_down(
        up_count=jnp.array(0),
        down_count=jnp.array(0),
        sites_filled=jnp.array(0),
        total_sites=4,
        spin_up=2,
        spin_down=2,
    )
    assert bool(jnp.all(mask))


# ---- Gumbel top-K ----


def test_gumbel_step_no_nan_with_invalid_children() -> None:
    key = jax.random.key(0)
    parent_log_prob = jnp.array([0.0, INVALID_LOG_PROB])
    parent_perturbed = jnp.array([0.0, INVALID_LOG_PROB])
    parent_valid = jnp.array([True, False])
    # second beam is padding; first beam forbids state 1.
    conditional = jnp.array([[jnp.log(0.6), -jnp.inf], [jnp.log(0.5), jnp.log(0.5)]])
    child_log_prob, child_perturbed, child_valid = gumbel_topk_step(
        parent_log_prob, parent_perturbed, parent_valid, conditional, key
    )
    assert not bool(jnp.any(jnp.isnan(child_perturbed)))
    assert not bool(jnp.any(jnp.isnan(child_log_prob)))
    # valid only where parent valid and conditional finite.
    assert bool(child_valid[0, 0])
    assert not bool(child_valid[0, 1])
    assert not bool(child_valid[1, 0])
    assert not bool(child_valid[1, 1])


def test_gumbel_step_perturbed_not_exceeding_parent() -> None:
    # The conditioned perturbed value must not exceed the parent's (upper bound).
    key = jax.random.key(1)
    parent_log_prob = jnp.array([0.0])
    parent_perturbed = jnp.array([-0.5])
    parent_valid = jnp.array([True])
    conditional = normalize_log_amplitude(jnp.array([[0.2, -0.3]]), axis=-1)
    _, child_perturbed, _ = gumbel_topk_step(parent_log_prob, parent_perturbed, parent_valid, conditional, key)
    finite = child_perturbed[jnp.isfinite(child_perturbed)]
    assert bool(jnp.all(finite <= parent_perturbed[0] + 1e-9))


def test_gumbel_step_child_log_prob_accumulates() -> None:
    key = jax.random.key(2)
    parent_log_prob = jnp.array([jnp.log(0.5)])
    parent_perturbed = jnp.array([0.0])
    parent_valid = jnp.array([True])
    conditional = jnp.array([[jnp.log(0.4), jnp.log(0.6)]])
    child_log_prob, _, _ = gumbel_topk_step(parent_log_prob, parent_perturbed, parent_valid, conditional, key)
    assert jnp.allclose(child_log_prob[0, 0], jnp.log(0.5) + jnp.log(0.4))
    assert jnp.allclose(child_log_prob[0, 1], jnp.log(0.5) + jnp.log(0.6))


# ---- sampling ----


def test_sample_step_respects_forbidden_states() -> None:
    key = jax.random.key(3)
    # State 0 forbidden -> all samples should be state 1.
    conditional = jnp.broadcast_to(jnp.array([-jnp.inf, 0.0]), (256, 2))
    samples = sample_step(conditional, key)
    assert bool(jnp.all(samples == 1))


def test_sample_step_determinism() -> None:
    key = jax.random.key(4)
    conditional = normalize_log_amplitude(jnp.broadcast_to(jnp.array([0.1, -0.2, 0.3]), (16, 3)), axis=-1)
    first = sample_step(conditional, key)
    second = sample_step(conditional, key)
    assert jnp.array_equal(first, second)


def test_sample_step_empirical_distribution() -> None:
    """Sampled frequencies approximate the Born distribution exp(2*log_amp)."""
    key = jax.random.key(5)
    # Fixed conditional over 3 states, normalised so probabilities are exp(2*x).
    conditional = normalize_log_amplitude(jnp.array([0.4, -0.1, 0.2]), axis=-1)
    target = jnp.exp(2.0 * conditional)
    batch = jnp.broadcast_to(conditional, (40000, 3))
    samples = sample_step(batch, key)
    counts = jnp.array([jnp.sum(samples == state) for state in range(3)])
    frequency = counts / jnp.sum(counts)
    assert jnp.max(jnp.abs(frequency - target)) < 0.01


def test_gumbel_step_all_children_invalid_when_parent_padding() -> None:
    """A padding parent yields only invalid children (no NaN, all sentinel)."""
    key = jax.random.key(6)
    parent_log_prob = jnp.array([INVALID_LOG_PROB])
    parent_perturbed = jnp.array([INVALID_LOG_PROB])
    parent_valid = jnp.array([False])
    conditional = normalize_log_amplitude(jnp.array([[0.5, 0.5]]), axis=-1)
    child_log_prob, child_perturbed, child_valid = gumbel_topk_step(
        parent_log_prob, parent_perturbed, parent_valid, conditional, key
    )
    assert not bool(jnp.any(child_valid))
    assert not bool(jnp.any(jnp.isnan(child_perturbed)))
    assert bool(jnp.all(child_perturbed == INVALID_LOG_PROB))
    assert bool(jnp.all(child_log_prob == INVALID_LOG_PROB))


def test_gumbel_step_invalid_children_sink_below_valid() -> None:
    """Invalid children always sort below valid ones (sentinel is very negative)."""
    key = jax.random.key(7)
    parent_log_prob = jnp.array([0.0, 0.0])
    parent_perturbed = jnp.array([0.0, 0.0])
    parent_valid = jnp.array([True, False])  # second beam is padding
    conditional = normalize_log_amplitude(jnp.array([[0.3, 0.7], [0.5, 0.5]]), axis=-1)
    _, child_perturbed, child_valid = gumbel_topk_step(
        parent_log_prob, parent_perturbed, parent_valid, conditional, key
    )
    flat_perturbed = child_perturbed.reshape(-1)
    flat_valid = child_valid.reshape(-1)
    valid_min = jnp.min(jnp.where(flat_valid, flat_perturbed, jnp.inf))
    invalid_max = jnp.max(jnp.where(flat_valid, -jnp.inf, flat_perturbed))
    assert float(invalid_max) < float(valid_min)
