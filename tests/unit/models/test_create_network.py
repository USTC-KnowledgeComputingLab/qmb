"""Tests for Model.create_network — model-coupled network construction.

``create_network(name, params, *, rngs)`` looks the name up in the model's
``network_dict``, deserialises ``params`` into the network config dataclass via
dacite, and calls ``cfg.create(model, rngs=rngs)``. These tests exercise that
dispatch path directly (the per-variant construction is covered separately in
each model's test file).
"""

from __future__ import annotations

import itertools

import dacite
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qmp.models.hubbard import Model, ModelConfig
from qmp.networks.transformers import WaveFunctionElectron as TransformersWaveFunctionElectron
from qmp.utility.bitspack import pack_int


def _all_configs(n_qubits: int) -> jax.Array:
    values = jnp.array(list(itertools.product([0, 1], repeat=n_qubits)), dtype=jnp.uint8)
    return pack_int(values, size=1)


@pytest.fixture
def model() -> Model:
    # 4 qubits, N=2 -> spin_up = spin_down = 1.
    return Model(ModelConfig(m=2, n=1, u=4.0))


# ---- dispatch by name ----


@pytest.mark.parametrize("name", ["mlp/u1u1", "mlp/u1", "transformers/u1u1", "transformers/u1"])
def test_create_network_dispatches_each_registered_name(model: Model, name: str) -> None:
    """Every registered network name builds a normalised wave function."""
    if name.startswith("transformers"):
        params: dict[str, object] = {
            "embedding_dim": 8,
            "heads_num": 2,
            "feed_forward_dim": 16,
            "depth": 1,
            "tail_hidden_dim": 8,
        }
    else:
        params = {"hidden_size": [8]}
    network = model.create_network(name, params, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_create_network_unknown_name_raises(model: Model) -> None:
    """An unregistered network name raises KeyError."""
    with pytest.raises(KeyError):
        model.create_network("does-not-exist", {}, rngs=nnx.Rngs(0))


# ---- params deserialisation via dacite ----


def test_create_network_empty_params_uses_defaults(model: Model) -> None:
    """Empty params → config dataclass defaults are used."""
    network = model.create_network("transformers/u1", {}, rngs=nnx.Rngs(0))
    # Default embedding_dim is 512 (see TransformersElectronConfig).
    assert isinstance(network, TransformersWaveFunctionElectron)
    assert network.embedding_dim == 512


def test_create_network_params_override_defaults(model: Model) -> None:
    """Provided params override the config defaults."""
    network = model.create_network(
        "transformers/u1",
        {"embedding_dim": 8, "heads_num": 2, "feed_forward_dim": 16, "depth": 1, "tail_hidden_dim": 8},
        rngs=nnx.Rngs(0),
    )
    assert isinstance(network, TransformersWaveFunctionElectron)
    assert network.embedding_dim == 8


def test_create_network_bad_param_type_raises(model: Model) -> None:
    """A param of the wrong type surfaces as a dacite error."""
    with pytest.raises(dacite.DaciteError):
        model.create_network("mlp/u1", {"ordering": "not-an-int"}, rngs=nnx.Rngs(0))


def test_create_network_unknown_param_ignored(model: Model) -> None:
    """Unknown params are ignored by dacite (lenient construction)."""
    network = model.create_network("mlp/u1", {"hidden_size": [8], "extra": "ignored"}, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


# ---- model metadata injection ----


def test_create_network_injects_model_electron_number(model: Model) -> None:
    """create_network derives conservation from model metadata: N=2 is enforced."""
    network = model.create_network("mlp/u1", {"hidden_size": [8]}, rngs=nnx.Rngs(0))
    values = jnp.array(list(itertools.product([0, 1], repeat=model.n_qubits)), dtype=jnp.uint8)
    psi = network(pack_int(values, size=1))
    assert jnp.all(jnp.abs(psi)[values.sum(axis=1) != 2] < 1e-12)


def test_create_network_prng_determinism(model: Model) -> None:
    """Same rngs seed → identical network parameters."""
    first = model.create_network("mlp/u1", {"hidden_size": [8]}, rngs=nnx.Rngs(7))
    second = model.create_network("mlp/u1", {"hidden_size": [8]}, rngs=nnx.Rngs(7))
    configs = _all_configs(model.n_qubits)
    assert jnp.allclose(first(configs), second(configs))


def test_create_network_result_is_sampleable(model: Model) -> None:
    """The built network is fully NetworkProto-conformant (generate / generate_unique work)."""
    network = model.create_network("mlp/u1", {"hidden_size": [8]}, rngs=nnx.Rngs(0))

    configs, psi, counts = network.generate(64, key=jax.random.key(0))
    assert int(jnp.sum(counts)) == 64
    assert jnp.allclose(psi, network(configs))

    unique_configs, unique_psi = network.generate_unique(6, key=jax.random.key(1))
    assert len(jnp.unique(unique_configs, axis=0)) == unique_configs.shape[0]
    assert jnp.allclose(unique_psi, network(unique_configs))
