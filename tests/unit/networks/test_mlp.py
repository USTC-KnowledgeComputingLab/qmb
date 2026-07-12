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


# ---- sampling correctness ----


def test_generate_empirical_frequency_matches_born() -> None:
    """Sampled frequencies approximate the Born distribution |psi|^2."""
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(16,), rngs=nnx.Rngs(20))
    all_configs = _all_configs(2, 4, 1)
    probability = jnp.abs(net(all_configs)) ** 2

    configs, _, counts = net.generate(40000, key=jax.random.key(0))
    frequency = counts / jnp.sum(counts)

    lookup = {tuple(row): float(probability[index]) for index, row in enumerate(all_configs.tolist())}
    for row, freq in zip(configs.tolist(), frequency.tolist(), strict=True):
        assert abs(freq - lookup[tuple(row)]) < 0.02


def test_generate_unique_is_exhaustive_when_beam_covers_space() -> None:
    """With beam width >= support size, generate_unique returns exactly the support set."""
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(16,), rngs=nnx.Rngs(21))
    all_configs = _all_configs(2, 4, 1)
    probability = jnp.abs(net(all_configs)) ** 2
    support = {tuple(row) for row, prob in zip(all_configs.tolist(), probability.tolist(), strict=True) if prob > 1e-12}

    configs, _ = net.generate_unique(6, key=jax.random.key(0))
    generated = {tuple(row) for row in configs.tolist()}
    assert generated == support


def test_generate_unique_caps_at_support_size() -> None:
    """Requesting more unique samples than exist returns exactly the support size."""
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), rngs=nnx.Rngs(22))
    configs, _ = net.generate_unique(50, key=jax.random.key(0))
    # Support size for C(4, 2) = 6.
    assert configs.shape[0] == 6
    assert len(jnp.unique(configs, axis=0)) == configs.shape[0]


def test_generate_determinism() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(8,), rngs=nnx.Rngs(23))
    first = net.generate(300, key=jax.random.key(7))
    second = net.generate(300, key=jax.random.key(7))
    assert jnp.array_equal(first[0], second[0])
    assert jnp.array_equal(first[2], second[2])


def test_generate_unique_different_keys_differ() -> None:
    """A near-uniform net gives different unique subsets for different keys."""
    net = WaveFunctionElectron(sites=6, electrons=3, hidden_size=(8,), rngs=nnx.Rngs(24))
    first, _ = net.generate_unique(3, key=jax.random.key(1))
    second, _ = net.generate_unique(3, key=jax.random.key(2))
    assert not jnp.array_equal(first, second)


# ---- conservation edge cases ----


def test_electron_zero_conserved() -> None:
    """electrons=0 puts all amplitude on the vacuum configuration."""
    net = WaveFunctionElectron(sites=3, electrons=0, hidden_size=(8,), rngs=nnx.Rngs(25))
    all_configs = _all_configs(2, 3, 1)
    probability = jnp.abs(net(all_configs)) ** 2
    assert jnp.allclose(probability[0], 1.0)  # index 0 == (0,0,0)
    assert jnp.allclose(jnp.sum(probability), 1.0)


def test_electron_full_conserved() -> None:
    """electrons=sites puts all amplitude on the fully occupied configuration."""
    net = WaveFunctionElectron(sites=3, electrons=3, hidden_size=(8,), rngs=nnx.Rngs(26))
    all_configs = _all_configs(2, 3, 1)
    probability = jnp.abs(net(all_configs)) ** 2
    assert jnp.allclose(probability[-1], 1.0)  # index -1 == (1,1,1)
    assert jnp.allclose(jnp.sum(probability), 1.0)


def test_updown_generate_unique_exhaustive() -> None:
    """UpDown generate_unique enumerates the full spin-resolved support."""
    net = WaveFunctionElectronUpDown(double_sites=6, spin_up=1, spin_down=1, hidden_size=(8,), rngs=nnx.Rngs(27))
    configs, _ = net.generate_unique(50, key=jax.random.key(0))
    # Support: C(3, 1) up * C(3, 1) down = 9.
    assert configs.shape[0] == 9
    values = unpack_int(configs, size=1, last_dim=6)
    up_count = values[:, 0] + values[:, 2] + values[:, 4]
    down_count = values[:, 1] + values[:, 3] + values[:, 5]
    assert jnp.all(up_count == 1)
    assert jnp.all(down_count == 1)


# ---- initialisation & gradients ----


def test_initial_state_is_real() -> None:
    """Zero-initialised output heads give a real initial wave function (phase 0)."""
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(16,), rngs=nnx.Rngs(28))
    psi = net(_all_configs(2, 4, 1))
    assert jnp.max(jnp.abs(jnp.imag(psi))) < 1e-12


def test_gradient_flows_through_call() -> None:
    """Gradients w.r.t. parameters are finite and non-zero (VMC trainability)."""
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(16,), rngs=nnx.Rngs(29))
    all_configs = _all_configs(2, 4, 1)
    graphdef, params = nnx.split(net, nnx.Param)

    def loss(params_state: nnx.State) -> jax.Array:
        model = nnx.merge(graphdef, params_state)
        psi = model(all_configs)
        return jnp.sum(jnp.real(psi) ** 2 + jnp.imag(psi) ** 2)

    grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert any(float(jnp.sum(jnp.abs(leaf))) > 0.0 for leaf in leaves)


# ---- amplitude invariances ----


def test_call_batch_invariance() -> None:
    """Amplitudes are independent of how configs are batched."""
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(16,), rngs=nnx.Rngs(30))
    all_configs = _all_configs(2, 4, 1)
    batched = net(all_configs)
    one_by_one = jnp.concatenate([net(all_configs[index : index + 1]) for index in range(all_configs.shape[0])])
    assert jnp.allclose(batched, one_by_one)


def test_call_jit_matches_eager() -> None:
    """Amplitudes under jit equal eager evaluation."""
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(16,), rngs=nnx.Rngs(31))
    all_configs = _all_configs(2, 4, 1)
    graphdef, state = nnx.split(net)
    eager = net(all_configs)
    jitted = jax.jit(lambda s, c: nnx.merge(graphdef, s)(c))(state, all_configs)
    assert jnp.allclose(eager, jitted)


def test_born_distribution_independent_of_phase() -> None:
    """Perturbing only the phase network leaves |psi|^2 unchanged but changes psi."""
    net = WaveFunctionElectron(sites=4, electrons=2, hidden_size=(16,), rngs=nnx.Rngs(32))
    all_configs = _all_configs(2, 4, 1)
    baseline = net(all_configs)

    graphdef, state = nnx.split(net)

    def bump_phase(path: tuple, leaf: jax.Array) -> jax.Array:
        if any("phase" in str(key) for key in path):
            return leaf + 1.0
        return leaf

    perturbed_state = jax.tree_util.tree_map_with_path(bump_phase, state)
    perturbed = nnx.merge(graphdef, perturbed_state)(all_configs)

    assert jnp.allclose(jnp.abs(baseline) ** 2, jnp.abs(perturbed) ** 2)
    assert jnp.max(jnp.abs(baseline - perturbed)) > 1e-6


def test_config_round_trip_identity() -> None:
    """decode(encode(config)) == config for the internal site-value mapping."""
    net = WaveFunctionElectronUpDown(double_sites=6, spin_up=1, spin_down=1, hidden_size=(8,), rngs=nnx.Rngs(33))
    configs, _ = net.generate_unique(9, key=jax.random.key(0))
    site_values = net._config_to_site_values(configs)
    round_tripped = net._site_values_to_config(site_values)
    assert jnp.array_equal(configs, round_tripped)


def test_generate_unique_subset_probability_bounded() -> None:
    """A strict subset of the support carries total probability < 1."""
    net = WaveFunctionElectron(sites=6, electrons=3, hidden_size=(8,), rngs=nnx.Rngs(34))
    _, psi = net.generate_unique(3, key=jax.random.key(0))
    total = float(jnp.sum(jnp.abs(psi) ** 2))
    assert 0.0 < total < 1.0 + 1e-9


def test_normal_qudit_physical_dim_four() -> None:
    """Normal variant normalises for a non-binary qudit (physical_dim=4)."""
    net = WaveFunctionNormal(sites=3, physical_dim=4, hidden_size=(8,), rngs=nnx.Rngs(35))
    psi = net(_all_configs(4, 3, 2))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)
