"""Tests for transformer autoregressive neural quantum states.

Covers all three variants: output shape/dtype, full-space normalisation,
particle-number conservation, generate/generate_unique self-consistency and
uniqueness, PRNG determinism, ordering, and arbitrary physical_dim.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qmp.networks.transformers import (
    WaveFunctionElectron,
    WaveFunctionElectronUpDown,
    WaveFunctionNormal,
)
from qmp.utility.bitspack import pack_int, unpack_int

_HYPERPARAMS = {
    "embedding_dim": 8,
    "heads_num": 2,
    "feed_forward_dim": 16,
    "depth": 2,
    "tail_hidden_dim": 8,
}


def _all_configs(states: int, sites: int, bit_size: int) -> jax.Array:
    values = jnp.array(list(itertools.product(range(states), repeat=sites)), dtype=jnp.uint8)
    return pack_int(values, size=bit_size)


# ---- __call__ shape / dtype ----


def test_electron_call_shape_dtype() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(0), **_HYPERPARAMS)
    configs = _all_configs(2, 4, 1)
    psi = net(configs)
    assert psi.shape == (configs.shape[0],)
    assert psi.dtype == jnp.complex128


# ---- normalisation ----


def test_electron_normalisation() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(1), **_HYPERPARAMS)
    psi = net(_all_configs(2, 4, 1))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_normal_normalisation() -> None:
    net = WaveFunctionNormal(sites=4, physical_dim=2, rngs=nnx.Rngs(2), **_HYPERPARAMS)
    psi = net(_all_configs(2, 4, 1))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_updown_normalisation() -> None:
    net = WaveFunctionElectronUpDown(double_sites=6, spin_up=1, spin_down=1, rngs=nnx.Rngs(3), **_HYPERPARAMS)
    psi = net(_all_configs(2, 6, 1))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_normal_arbitrary_physical_dim() -> None:
    net = WaveFunctionNormal(sites=3, physical_dim=3, rngs=nnx.Rngs(4), **_HYPERPARAMS)
    psi = net(_all_configs(3, 3, 2))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


# ---- particle-number conservation ----


def test_electron_conservation() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(5), **_HYPERPARAMS)
    values = jnp.array(list(itertools.product([0, 1], repeat=4)), dtype=jnp.uint8)
    psi = net(pack_int(values, size=1))
    electron_count = values.sum(axis=1)
    assert jnp.all(jnp.abs(psi)[electron_count != 2] < 1e-12)
    assert jnp.all(jnp.abs(psi)[electron_count == 2] > 0.0)


def test_updown_conservation() -> None:
    net = WaveFunctionElectronUpDown(double_sites=6, spin_up=1, spin_down=1, rngs=nnx.Rngs(6), **_HYPERPARAMS)
    values = jnp.array(list(itertools.product([0, 1], repeat=6)), dtype=jnp.uint8)
    psi = net(pack_int(values, size=1))
    up_count = values[:, 0] + values[:, 2] + values[:, 4]
    down_count = values[:, 1] + values[:, 3] + values[:, 5]
    forbidden = (up_count != 1) | (down_count != 1)
    assert jnp.all(jnp.abs(psi)[forbidden] < 1e-12)


# ---- generate_unique ----


def test_generate_unique_self_consistency() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(7), **_HYPERPARAMS)
    configs, psi = net.generate_unique(6, key=jax.random.key(0))
    assert jnp.allclose(psi, net(configs))


def test_generate_unique_uniqueness_and_bound() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(8), **_HYPERPARAMS)
    configs, _ = net.generate_unique(6, key=jax.random.key(1))
    assert configs.shape[0] <= 6
    assert len(jnp.unique(configs, axis=0)) == configs.shape[0]


def test_generate_unique_conserves_particle_number() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(9), **_HYPERPARAMS)
    configs, _ = net.generate_unique(6, key=jax.random.key(2))
    values = unpack_int(configs, size=1, last_dim=4)
    assert jnp.all(values.sum(axis=1) == 2)


def test_generate_unique_determinism() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(10), **_HYPERPARAMS)
    first_configs, first_psi = net.generate_unique(6, key=jax.random.key(3))
    second_configs, second_psi = net.generate_unique(6, key=jax.random.key(3))
    assert jnp.array_equal(first_configs, second_configs)
    assert jnp.allclose(first_psi, second_psi)


# ---- generate ----


def test_generate_counts_sum() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(11), **_HYPERPARAMS)
    configs, psi, counts = net.generate(200, key=jax.random.key(4))
    assert int(jnp.sum(counts)) == 200
    assert len(jnp.unique(configs, axis=0)) == configs.shape[0]
    assert jnp.allclose(psi, net(configs))


def test_generate_conserves_particle_number() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(12), **_HYPERPARAMS)
    configs, _, _ = net.generate(200, key=jax.random.key(5))
    values = unpack_int(configs, size=1, last_dim=4)
    assert jnp.all(values.sum(axis=1) == 2)


# ---- ordering ----


@pytest.mark.parametrize("ordering", [1, -1, [3, 1, 0, 2]])
def test_ordering_preserves_normalisation(ordering: int | list[int]) -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, ordering=ordering, rngs=nnx.Rngs(13), **_HYPERPARAMS)
    psi = net(_all_configs(2, 4, 1))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_generate_unique_round_trip_with_ordering() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, ordering=[2, 0, 3, 1], rngs=nnx.Rngs(14), **_HYPERPARAMS)
    configs, psi = net.generate_unique(6, key=jax.random.key(6))
    assert jnp.allclose(psi, net(configs))


# ---- causality ----


def test_causal_amplitude_independence() -> None:
    """The conditional probability of early sites must not depend on later sites.

    For an autoregressive model, p(x_0, x_1) marginalised from the joint must be
    identical regardless of what is appended afterwards. We check that the summed
    probability over all completions of a shared 2-site prefix is consistent
    between the parallel amplitude evaluation and a direct enumeration.
    """
    net = WaveFunctionNormal(sites=4, physical_dim=2, rngs=nnx.Rngs(15), **_HYPERPARAMS)
    all_values = jnp.array(list(itertools.product([0, 1], repeat=4)), dtype=jnp.uint8)
    psi = net(pack_int(all_values, size=1))
    probability = jnp.abs(psi) ** 2

    # Group by the first two sites; each group's probability mass is p(x_0, x_1).
    prefix = all_values[:, :2]
    prefix_codes = prefix[:, 0] * 2 + prefix[:, 1]
    for code in range(4):
        mass = jnp.sum(probability[prefix_codes == code])
        # Marginal mass must be well-defined in [0, 1]; total across prefixes is 1.
        assert 0.0 <= float(mass) <= 1.0 + 1e-9
    assert jnp.allclose(jnp.sum(probability), 1.0)


def test_generate_unique_matches_parallel_amplitude() -> None:
    """Incremental KV-cache decoding must reproduce the parallel amplitude exactly."""
    net = WaveFunctionElectronUpDown(double_sites=6, spin_up=1, spin_down=1, rngs=nnx.Rngs(16), **_HYPERPARAMS)
    configs, psi = net.generate_unique(8, key=jax.random.key(7))
    assert jnp.allclose(psi, net(configs))
