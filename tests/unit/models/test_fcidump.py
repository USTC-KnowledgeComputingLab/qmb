"""Tests for the FCIDUMP model."""

from __future__ import annotations

import gzip
import itertools
import pathlib
import pickle

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import qmp.models.fcidump as fcidump_module
from qmp.models._model import model_dict
from qmp.models.fcidump import (
    MlpElectronConfig,
    MlpUpDownConfig,
    Model,
    ModelConfig,
    TransformersElectronConfig,
    TransformersUpDownConfig,
    read_fcidump,
)
from qmp.networks.transformers import WaveFunctionElectron as TransformersWaveFunctionElectron
from qmp.utility.bitspack import pack_int, unpack_int

_FCIDUMP_H2 = """&FCI NORB=2,NELEC=2,MS2=0,
ORBSYM=1,1,
ISYM=1,
&END
  0.6746000000  1  1  1  1
 -1.2524000000  1  1  0  0
 -0.4759000000  2  2  0  0
  0.6976000000  2  2  2  2
  0.7137539936  0  0  0  0
"""


def _write_fcidump(tmp_path) -> str:
    path = tmp_path / "FCIDUMP"
    path.write_text(_FCIDUMP_H2, encoding="utf-8")
    return str(path)


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect the fcidump cache dir into tmp_path so tests have no side effects."""
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(fcidump_module, "_cache_dir", lambda: cache_root / "models" / "fcidump")


def test_fcidump_registered() -> None:
    """FCIDUMP model registers itself."""
    assert model_dict["fcidump"] is Model


def test_fcidump_metadata(tmp_path) -> None:
    """Metadata read from the FCIDUMP header."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    assert model.n_qubits == 4
    assert model.n_electrons == 2
    assert model.n_spins == 0


def test_fcidump_ref_energy_default(tmp_path) -> None:
    """ref_energy defaults to 0.0 (does NOT read FCIDUMP.yaml)."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    assert model.ref_energy == 0.0


def test_fcidump_ref_energy_explicit(tmp_path) -> None:
    """ref_energy is taken from config when provided."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path), ref_energy=-1.137))
    assert model.ref_energy == -1.137


def test_fcidump_cache_hit(tmp_path) -> None:
    """Second construction loads the Hamiltonian from cache and matches."""
    path = _write_fcidump(tmp_path)
    first = Model(ModelConfig(model_path=path))
    second = Model(ModelConfig(model_path=path))  # cache hit path
    diag_a = first.compute_diagonal_within_subspace(jnp.array([[0b00000011]], dtype=jnp.uint8))
    diag_b = second.compute_diagonal_within_subspace(jnp.array([[0b00000011]], dtype=jnp.uint8))
    assert jnp.allclose(diag_a, diag_b)


def test_fcidump_diagonal_matches_hand_computation(tmp_path) -> None:
    """The parsed Hamiltonian yields the expected diagonal for the closed-shell config.

    Both electrons occupy spatial orbital 0 (spin-orbitals 0 and 1, config 0b0011).
    Diagonal = 2 * h_00 + <00|00> = 2*(-1.2524) + 0.6746 = -1.8302.
    """
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    config = jnp.array([[0b00000011]], dtype=jnp.uint8)
    diagonal = model.compute_diagonal_within_subspace(config)
    assert float(diagonal[0, 0]) == pytest.approx(-1.8302, abs=1e-4)
    assert float(diagonal[0, 1]) == pytest.approx(0.0, abs=1e-6)


def test_fcidump_empty_config_diagonal_zero(tmp_path) -> None:
    """The vacuum config has zero diagonal (the constant term is not a fermionic operator)."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    config = jnp.array([[0b00000000]], dtype=jnp.uint8)
    diagonal = model.compute_diagonal_within_subspace(config)
    assert float(diagonal[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_fcidump_cache_file_written(tmp_path) -> None:
    """Constructing a model writes exactly one .pkl cache file into models/fcidump."""
    Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    cache_files = list(fcidump_module._cache_dir().glob("*.pkl"))
    assert len(cache_files) == 1


def test_fcidump_cache_content_matches_fresh_parse(tmp_path) -> None:
    """The pickled cache holds exactly the Hamiltonian dict of a fresh parse."""
    path = _write_fcidump(tmp_path)
    Model(ModelConfig(model_path=path))
    cache_file = next(fcidump_module._cache_dir().glob("*.pkl"))
    with open(cache_file, "rb") as file:
        cached = pickle.load(file)
    (_, fresh) = read_fcidump(pathlib.Path(path))
    assert cached == fresh


def test_fcidump_distinct_content_distinct_cache(tmp_path) -> None:
    """Different file contents produce different cache keys (no collision)."""
    path_a = tmp_path / "a"
    path_b = tmp_path / "b"
    path_a.write_text(_FCIDUMP_H2, encoding="utf-8")
    path_b.write_text(_FCIDUMP_H2.replace("-1.2524000000", "-2.0000000000"), encoding="utf-8")
    Model(ModelConfig(model_path=str(path_a)))
    Model(ModelConfig(model_path=str(path_b)))
    assert len(list(fcidump_module._cache_dir().glob("*.pkl"))) == 2


def test_read_fcidump_headonly(tmp_path) -> None:
    """headonly parse returns metadata and an empty Hamiltonian dict."""
    path = pathlib.Path(_write_fcidump(tmp_path))
    (n_orbit, n_electron, n_spin), hamiltonian = read_fcidump(path, headonly=True)
    assert (n_orbit, n_electron, n_spin) == (2, 2, 0)
    assert hamiltonian == {}


def test_read_fcidump_full_has_terms(tmp_path) -> None:
    """Full parse returns a non-empty Hamiltonian dict with complex coefficients."""
    path = pathlib.Path(_write_fcidump(tmp_path))
    (n_orbit, _, _), hamiltonian = read_fcidump(path)
    assert n_orbit == 2
    assert len(hamiltonian) > 0
    assert all(isinstance(value, complex) for value in hamiltonian.values())


def test_read_fcidump_gzip(tmp_path) -> None:
    """A .gz FCIDUMP file is transparently decompressed and parsed identically."""
    plain = pathlib.Path(_write_fcidump(tmp_path))
    gz_path = tmp_path / "FCIDUMP.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as file:
        file.write(_FCIDUMP_H2)
    (_, plain_dict) = read_fcidump(plain)
    (_, gz_dict) = read_fcidump(gz_path)
    assert gz_dict == plain_dict


def test_read_fcidump_psi4_one_electron_ignored(tmp_path) -> None:
    """Psi4 non-standard one-index integral lines (i, -1, -1, -1) are silently skipped."""
    header = "&FCI NORB=2,NELEC=2,MS2=0,\n&END\n"
    with_line = tmp_path / "FCIDUMP_psi4"
    without_line = tmp_path / "FCIDUMP_ref"
    with_line.write_text(header + " -1.25 1 1 0 0\n  0.9 1 0 0 0\n  0.71 0 0 0 0\n", encoding="utf-8")
    without_line.write_text(header + " -1.25 1 1 0 0\n  0.71 0 0 0 0\n", encoding="utf-8")
    (_, with_psi4) = read_fcidump(with_line)
    (_, without_psi4) = read_fcidump(without_line)
    # the extra Psi4 one-electron line must not change the resulting Hamiltonian
    assert with_psi4 == without_psi4


def test_read_fcidump_invalid_format_raises(tmp_path) -> None:
    """A three-index integral line (invalid standard FCIDUMP) raises ValueError."""
    bad = tmp_path / "FCIDUMP_bad"
    bad.write_text(
        "&FCI NORB=2,NELEC=2,MS2=0,\n&END\n  0.5  1  2  3  0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid FCIDUMP format"):
        read_fcidump(bad)


# ---- network_dict construction ----

_SMALL_NETWORK_PARAMS = {
    "embedding_dim": 8,
    "heads_num": 2,
    "feed_forward_dim": 16,
    "depth": 1,
    "tail_hidden_dim": 8,
}


def test_fcidump_network_dict_keys() -> None:
    """FCIDUMP registers the four particle-conserving network configs."""
    assert Model.network_dict == {
        "mlp/u1u1": MlpUpDownConfig,
        "mlp/u1": MlpElectronConfig,
        "transformers/u1u1": TransformersUpDownConfig,
        "transformers/u1": TransformersElectronConfig,
    }


def _all_configs(n_qubits: int) -> jnp.ndarray:
    values = jnp.array(list(itertools.product([0, 1], repeat=n_qubits)), dtype=jnp.uint8)
    return pack_int(values, size=1)


def test_fcidump_mlp_u1u1_construction(tmp_path) -> None:
    """mlp/u1u1 builds a normalised spin-resolved wave function for the H2 FCIDUMP."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))  # n_qubits=4, N=2, MS2=0
    network = MlpUpDownConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_fcidump_mlp_u1_construction(tmp_path) -> None:
    """mlp/u1 builds a normalised total-electron wave function."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    network = MlpElectronConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_fcidump_transformers_u1u1_construction(tmp_path) -> None:
    """transformers/u1u1 builds a normalised wave function."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    network = TransformersUpDownConfig(**_SMALL_NETWORK_PARAMS).create(model, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_fcidump_transformers_u1_construction(tmp_path) -> None:
    """transformers/u1 builds a normalised wave function."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    network = TransformersElectronConfig(**_SMALL_NETWORK_PARAMS).create(model, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_fcidump_mlp_u1u1_conservation(tmp_path) -> None:
    """mlp/u1u1 enforces spin-resolved conservation (H2: spin_up=spin_down=1)."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    network = MlpUpDownConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    values = jnp.array(list(itertools.product([0, 1], repeat=model.n_qubits)), dtype=jnp.uint8)
    up = values[:, 0] + values[:, 2]
    down = values[:, 1] + values[:, 3]
    assert jnp.all(jnp.abs(psi)[(up != 1) | (down != 1)] < 1e-12)


def test_fcidump_mlp_u1_conservation(tmp_path) -> None:
    """mlp/u1 enforces total-electron conservation (H2: N=2)."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    network = MlpElectronConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    values = jnp.array(list(itertools.product([0, 1], repeat=model.n_qubits)), dtype=jnp.uint8)
    assert jnp.all(jnp.abs(psi)[values.sum(axis=1) != 2] < 1e-12)


def test_fcidump_network_generate_unique(tmp_path) -> None:
    """generate_unique yields unique conserving configs consistent with __call__."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    network = MlpElectronConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(0))
    configs, psi = network.generate_unique(6, key=jax.random.key(0))
    assert len(jnp.unique(configs, axis=0)) == configs.shape[0]
    assert jnp.allclose(psi, network(configs))
    values = unpack_int(configs, size=1, last_dim=model.n_qubits)
    assert jnp.all(values.sum(axis=1) == 2)


def test_fcidump_network_prng_determinism(tmp_path) -> None:
    """Same rngs seed builds identical networks."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    first = MlpElectronConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(3))
    second = MlpElectronConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(3))
    configs = _all_configs(model.n_qubits)
    assert jnp.allclose(first(configs), second(configs))


def test_fcidump_config_fields_passed_to_network(tmp_path) -> None:
    """Transformer config hyperparameters propagate to the built network."""
    model = Model(ModelConfig(model_path=_write_fcidump(tmp_path)))
    network = TransformersElectronConfig(embedding_dim=16, depth=2, heads_num=4, tail_hidden_dim=8).create(
        model, rngs=nnx.Rngs(0)
    )
    assert isinstance(network, TransformersWaveFunctionElectron)
    assert network.embedding_dim == 16
