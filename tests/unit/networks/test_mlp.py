"""Tests for MLP autoregressive neural quantum states.

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

from qmp.networks.mlp import (
    WaveFunctionElectron,
    WaveFunctionElectronUpDown,
    WaveFunctionNormal,
)
from qmp.utility.bitspack import pack_int, unpack_int


def _all_configs(states: int, sites: int, bit_size: int) -> jax.Array:
    """Bit-packed configs enumerating the full Hilbert space."""
    values = jnp.array(list(itertools.product(range(states), repeat=sites)), dtype=jnp.uint8)
    return pack_int(values, size=bit_size)


# ---- __call__ shape / dtype ----


def test_electron_call_shape_dtype() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), rngs=nnx.Rngs(0))
    configs = _all_configs(2, 4, 1)
    psi = net(configs)
    assert psi.shape == (configs.shape[0],)
    assert psi.dtype == jnp.complex128


# ---- normalisation over the full Hilbert space ----


def test_electron_normalisation() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), rngs=nnx.Rngs(1))
    psi = net(_all_configs(2, 4, 1))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_normal_normalisation() -> None:
    net = WaveFunctionNormal(sites=4, physical_dim=2, hidden_size=(8,), rngs=nnx.Rngs(2))
    psi = net(_all_configs(2, 4, 1))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_updown_normalisation() -> None:
    net = WaveFunctionElectronUpDown(double_sites=6, spin_up=1, spin_down=1, hidden_size=(8,), rngs=nnx.Rngs(3))
    psi = net(_all_configs(2, 6, 1))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_normal_arbitrary_physical_dim() -> None:
    net = WaveFunctionNormal(sites=3, physical_dim=3, hidden_size=(8,), rngs=nnx.Rngs(4))
    psi = net(_all_configs(3, 3, 2))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


# ---- particle-number conservation ----


def test_electron_conservation() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), rngs=nnx.Rngs(5))
    values = jnp.array(list(itertools.product([0, 1], repeat=4)), dtype=jnp.uint8)
    psi = net(pack_int(values, size=1))
    electron_count = values.sum(axis=1)
    assert jnp.all(jnp.abs(psi)[electron_count != 2] < 1e-12)
    assert jnp.all(jnp.abs(psi)[electron_count == 2] > 0.0)


def test_updown_conservation() -> None:
    net = WaveFunctionElectronUpDown(double_sites=6, spin_up=1, spin_down=1, hidden_size=(8,), rngs=nnx.Rngs(6))
    values = jnp.array(list(itertools.product([0, 1], repeat=6)), dtype=jnp.uint8)
    psi = net(pack_int(values, size=1))
    up_count = values[:, 0] + values[:, 2] + values[:, 4]
    down_count = values[:, 1] + values[:, 3] + values[:, 5]
    forbidden = (up_count != 1) | (down_count != 1)
    assert jnp.all(jnp.abs(psi)[forbidden] < 1e-12)


# ---- generate_unique ----


def test_generate_unique_self_consistency() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), rngs=nnx.Rngs(7))
    configs, psi = net.generate_unique(6, key=jax.random.key(0))
    assert jnp.allclose(psi, net(configs))


def test_generate_unique_uniqueness_and_bound() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), rngs=nnx.Rngs(8))
    configs, _ = net.generate_unique(6, key=jax.random.key(1))
    assert configs.shape[0] <= 6
    assert len(jnp.unique(configs, axis=0)) == configs.shape[0]


def test_generate_unique_conserves_particle_number() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), rngs=nnx.Rngs(9))
    configs, _ = net.generate_unique(6, key=jax.random.key(2))
    values = unpack_int(configs, size=1, last_dim=4)
    assert jnp.all(values.sum(axis=1) == 2)


def test_generate_unique_determinism() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), rngs=nnx.Rngs(10))
    first_configs, first_psi = net.generate_unique(6, key=jax.random.key(3))
    second_configs, second_psi = net.generate_unique(6, key=jax.random.key(3))
    assert jnp.array_equal(first_configs, second_configs)
    assert jnp.allclose(first_psi, second_psi)


# ---- generate ----


def test_generate_counts_sum() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), rngs=nnx.Rngs(11))
    configs, psi, counts = net.generate(200, key=jax.random.key(4))
    assert int(jnp.sum(counts)) == 200
    assert len(jnp.unique(configs, axis=0)) == configs.shape[0]
    assert jnp.allclose(psi, net(configs))


def test_generate_conserves_particle_number() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), rngs=nnx.Rngs(12))
    configs, _, _ = net.generate(200, key=jax.random.key(5))
    values = unpack_int(configs, size=1, last_dim=4)
    assert jnp.all(values.sum(axis=1) == 2)


# ---- ordering ----


@pytest.mark.parametrize("ordering", [1, -1, [3, 1, 0, 2]])
def test_ordering_preserves_normalisation(ordering: int | list[int]) -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), ordering=ordering, rngs=nnx.Rngs(13))
    psi = net(_all_configs(2, 4, 1))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_generate_unique_round_trip_with_ordering() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), ordering=[2, 0, 3, 1], rngs=nnx.Rngs(14))
    configs, psi = net.generate_unique(6, key=jax.random.key(6))
    assert jnp.allclose(psi, net(configs))
