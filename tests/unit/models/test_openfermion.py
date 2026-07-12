"""Tests for the OpenFermion model."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import openfermion
import pytest
from flax import nnx

from qmp.models._model import model_dict
from qmp.models.openfermion import (
    MlpElectronConfig,
    MlpUpDownConfig,
    Model,
    ModelConfig,
    TransformersElectronConfig,
    TransformersUpDownConfig,
)
from qmp.utility.bitspack import pack_int, unpack_int


@pytest.fixture
def h2_molecule_file(tmp_path) -> str:
    """Build a minimal H2 MolecularData file and return its path (without extension)."""
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
    molecule = openfermion.MolecularData(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        filename=str(tmp_path / "h2"),
    )
    molecule.n_orbitals = 2
    molecule.n_qubits = 4
    molecule.n_electrons = 2
    molecule.fci_energy = -1.137
    molecule.nuclear_repulsion = 0.7137539936
    molecule.one_body_integrals = np.array([[-1.2524, 0.0], [0.0, -0.4759]])
    molecule.two_body_integrals = np.zeros((2, 2, 2, 2))
    molecule.two_body_integrals[0, 0, 0, 0] = 0.6746
    molecule.two_body_integrals[1, 1, 1, 1] = 0.6976
    molecule.save()
    return str(tmp_path / "h2")


def test_openfermion_registered() -> None:
    """OpenFermion model registers itself."""
    assert model_dict["openfermion"] is Model


def test_openfermion_metadata(h2_molecule_file: str) -> None:
    """Metadata (n_qubits, n_electrons, n_spins, ref_energy) read correctly."""
    model = Model(ModelConfig(model_path=h2_molecule_file))
    assert model.n_qubits == 4
    assert model.n_electrons == 2
    assert model.n_spins == 0  # multiplicity 1 -> 2S = 0
    assert model.ref_energy == pytest.approx(-1.137)


def test_openfermion_spin_from_multiplicity(tmp_path) -> None:
    """n_spins = multiplicity - 1 (triplet -> 2)."""
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
    molecule = openfermion.MolecularData(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=3,
        charge=0,
        filename=str(tmp_path / "h2_triplet"),
    )
    molecule.n_orbitals = 2
    molecule.n_qubits = 4
    molecule.n_electrons = 2
    molecule.fci_energy = -1.0
    molecule.nuclear_repulsion = 0.7137539936
    molecule.one_body_integrals = np.array([[-1.2524, 0.0], [0.0, -0.4759]])
    molecule.two_body_integrals = np.zeros((2, 2, 2, 2))
    molecule.save()
    model = Model(ModelConfig(model_path=str(tmp_path / "h2_triplet")))
    assert model.n_spins == 2


def test_openfermion_missing_fci_energy_raises(tmp_path) -> None:
    """A MolecularData without fci_energy raises a clear ValueError."""
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
    molecule = openfermion.MolecularData(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        filename=str(tmp_path / "h2_nofci"),
    )
    molecule.n_orbitals = 2
    molecule.n_qubits = 4
    molecule.n_electrons = 2
    molecule.nuclear_repulsion = 0.7137539936
    molecule.one_body_integrals = np.array([[-1.2524, 0.0], [0.0, -0.4759]])
    molecule.two_body_integrals = np.zeros((2, 2, 2, 2))
    molecule.save()
    with pytest.raises(ValueError, match="missing n_qubits or fci_energy"):
        Model(ModelConfig(model_path=str(tmp_path / "h2_nofci")))


def test_openfermion_forwards_diagonal(h2_molecule_file: str) -> None:
    """Operator calls forward to the wrapped FermiHamiltonian and return arrays."""
    model = Model(ModelConfig(model_path=h2_molecule_file))
    configs = jnp.array([[0b00000011]], dtype=jnp.uint8)  # lowest two spin-orbitals filled
    diagonal = model.compute_diagonal_within_subspace(configs)
    assert diagonal.shape == (1, 2)


# ---- network_dict construction ----

_SMALL_NETWORK_PARAMS = {
    "embedding_dim": 8,
    "heads_num": 2,
    "feed_forward_dim": 16,
    "depth": 1,
    "tail_hidden_dim": 8,
}


def _all_configs(n_qubits: int) -> jnp.ndarray:
    values = jnp.array(list(itertools.product([0, 1], repeat=n_qubits)), dtype=jnp.uint8)
    return pack_int(values, size=1)


def test_openfermion_network_dict_keys() -> None:
    """OpenFermion registers the four particle-conserving network configs."""
    assert Model.network_dict == {
        "mlp/u1u1": MlpUpDownConfig,
        "mlp/u1": MlpElectronConfig,
        "transformers/u1u1": TransformersUpDownConfig,
        "transformers/u1": TransformersElectronConfig,
    }


def test_openfermion_mlp_u1u1_construction(h2_molecule_file: str) -> None:
    """mlp/u1u1 builds a normalised spin-resolved wave function for H2."""
    model = Model(ModelConfig(model_path=h2_molecule_file))  # n_qubits=4, N=2, multiplicity 1
    network = MlpUpDownConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_openfermion_mlp_u1_construction(h2_molecule_file: str) -> None:
    """mlp/u1 builds a normalised total-electron wave function."""
    model = Model(ModelConfig(model_path=h2_molecule_file))
    network = MlpElectronConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_openfermion_transformers_u1u1_construction(h2_molecule_file: str) -> None:
    """transformers/u1u1 builds a normalised wave function."""
    model = Model(ModelConfig(model_path=h2_molecule_file))
    network = TransformersUpDownConfig(**_SMALL_NETWORK_PARAMS).create(model, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_openfermion_transformers_u1_construction(h2_molecule_file: str) -> None:
    """transformers/u1 builds a normalised wave function."""
    model = Model(ModelConfig(model_path=h2_molecule_file))
    network = TransformersElectronConfig(**_SMALL_NETWORK_PARAMS).create(model, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0)


def test_openfermion_mlp_u1u1_conservation(h2_molecule_file: str) -> None:
    """mlp/u1u1 enforces spin-resolved conservation for H2 (spin_up=spin_down=1)."""
    model = Model(ModelConfig(model_path=h2_molecule_file))
    network = MlpUpDownConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    values = jnp.array(list(itertools.product([0, 1], repeat=model.n_qubits)), dtype=jnp.uint8)
    up = values[:, 0] + values[:, 2]
    down = values[:, 1] + values[:, 3]
    assert jnp.all(jnp.abs(psi)[(up != 1) | (down != 1)] < 1e-12)


def test_openfermion_mlp_u1_conservation(h2_molecule_file: str) -> None:
    """mlp/u1 enforces total-electron conservation for H2 (N=2)."""
    model = Model(ModelConfig(model_path=h2_molecule_file))
    network = MlpElectronConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(0))
    psi = network(_all_configs(model.n_qubits))
    values = jnp.array(list(itertools.product([0, 1], repeat=model.n_qubits)), dtype=jnp.uint8)
    assert jnp.all(jnp.abs(psi)[values.sum(axis=1) != 2] < 1e-12)


def test_openfermion_network_generate_unique(h2_molecule_file: str) -> None:
    """generate_unique yields unique conserving configs consistent with __call__."""
    model = Model(ModelConfig(model_path=h2_molecule_file))
    network = MlpElectronConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(0))
    configs, psi = network.generate_unique(6, key=jax.random.key(0))
    assert len(jnp.unique(configs, axis=0)) == configs.shape[0]
    assert jnp.allclose(psi, network(configs))
    values = unpack_int(configs, size=1, last_dim=model.n_qubits)
    assert jnp.all(values.sum(axis=1) == 2)


def test_openfermion_network_prng_determinism(h2_molecule_file: str) -> None:
    """Same rngs seed builds identical networks."""
    model = Model(ModelConfig(model_path=h2_molecule_file))
    first = MlpElectronConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(5))
    second = MlpElectronConfig(hidden_size=(8,)).create(model, rngs=nnx.Rngs(5))
    configs = _all_configs(model.n_qubits)
    assert jnp.allclose(first(configs), second(configs))
