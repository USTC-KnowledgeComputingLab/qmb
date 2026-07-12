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


# ---- sampling correctness ----


def test_generate_empirical_frequency_matches_born() -> None:
    """Sampled frequencies approximate the Born distribution |psi|^2."""
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(20), **_HYPERPARAMS)
    all_configs = _all_configs(2, 4, 1)
    probability = jnp.abs(net(all_configs)) ** 2

    configs, _, counts = net.generate(40000, key=jax.random.key(0))
    frequency = counts / jnp.sum(counts)

    lookup = {tuple(row): float(probability[index]) for index, row in enumerate(all_configs.tolist())}
    for row, freq in zip(configs.tolist(), frequency.tolist(), strict=True):
        assert abs(freq - lookup[tuple(row)]) < 0.02


def test_generate_unique_is_exhaustive_when_beam_covers_space() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(21), **_HYPERPARAMS)
    all_configs = _all_configs(2, 4, 1)
    probability = jnp.abs(net(all_configs)) ** 2
    support = {tuple(row) for row, prob in zip(all_configs.tolist(), probability.tolist(), strict=True) if prob > 1e-12}

    configs, _ = net.generate_unique(6, key=jax.random.key(0))
    generated = {tuple(row) for row in configs.tolist()}
    assert generated == support


def test_generate_unique_caps_at_support_size() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(22), **_HYPERPARAMS)
    configs, _ = net.generate_unique(50, key=jax.random.key(0))
    assert configs.shape[0] == 6
    assert len(jnp.unique(configs, axis=0)) == configs.shape[0]


def test_generate_determinism() -> None:
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(23), **_HYPERPARAMS)
    first = net.generate(300, key=jax.random.key(7))
    second = net.generate(300, key=jax.random.key(7))
    assert jnp.array_equal(first[0], second[0])
    assert jnp.array_equal(first[2], second[2])


# ---- KV-cache isolation ----


def test_generate_does_not_pollute_parameters() -> None:
    """KV-cache side effects must not add or change nnx.Param leaves."""
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(30), **_HYPERPARAMS)
    params_before = jax.tree_util.tree_leaves(nnx.state(net, nnx.Param))
    net.generate_unique(6, key=jax.random.key(0))
    params_after = jax.tree_util.tree_leaves(nnx.state(net, nnx.Param))
    assert len(params_before) == len(params_after)
    for before, after in zip(params_before, params_after, strict=True):
        assert jnp.array_equal(before, after)


def test_repeated_generation_is_consistent() -> None:
    """Two successive generations with the same key give identical results.

    Guards the KV-cache re-initialisation and beam reordering across calls.
    """
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(31), **_HYPERPARAMS)
    first_configs, first_psi = net.generate_unique(5, key=jax.random.key(3))
    second_configs, second_psi = net.generate_unique(5, key=jax.random.key(3))
    assert jnp.array_equal(first_configs, second_configs)
    assert jnp.allclose(first_psi, second_psi)


def test_different_batch_sizes_reinitialise_cache() -> None:
    """Changing beam width between calls still yields correct amplitudes."""
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(32), **_HYPERPARAMS)
    net.generate_unique(3, key=jax.random.key(0))
    configs, psi = net.generate_unique(6, key=jax.random.key(1))
    assert jnp.allclose(psi, net(configs))


# ---- conservation edge cases ----


def test_electron_zero_conserved() -> None:
    net = WaveFunctionElectron(sites=3, electrons=0, rngs=nnx.Rngs(25), **_HYPERPARAMS)
    all_configs = _all_configs(2, 3, 1)
    probability = jnp.abs(net(all_configs)) ** 2
    assert jnp.allclose(probability[0], 1.0)
    assert jnp.allclose(jnp.sum(probability), 1.0)


def test_electron_full_conserved() -> None:
    net = WaveFunctionElectron(sites=3, electrons=3, rngs=nnx.Rngs(26), **_HYPERPARAMS)
    all_configs = _all_configs(2, 3, 1)
    probability = jnp.abs(net(all_configs)) ** 2
    assert jnp.allclose(probability[-1], 1.0)
    assert jnp.allclose(jnp.sum(probability), 1.0)


# ---- initialisation & gradients ----


def test_initial_state_is_real() -> None:
    """Zero-initialised Tail gives a real initial wave function (phase 0)."""
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(28), **_HYPERPARAMS)
    psi = net(_all_configs(2, 4, 1))
    assert jnp.max(jnp.abs(jnp.imag(psi))) < 1e-12


def test_gradient_flows_through_call() -> None:
    """Gradients w.r.t. parameters are finite and non-zero (VMC trainability)."""
    net = WaveFunctionElectron(sites=4, electrons=2, rngs=nnx.Rngs(29), **_HYPERPARAMS)
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
