"""Tests for the OpenFermion model."""

from __future__ import annotations

import numpy as np
import openfermion
import pytest

from qmp.models._model import model_dict


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
    from qmp.models.openfermion import Model

    assert model_dict["openfermion"] is Model


def test_openfermion_metadata(h2_molecule_file: str) -> None:
    """Metadata (n_qubits, n_electrons, n_spins, ref_energy) read correctly."""
    from qmp.models.openfermion import Model, ModelConfig

    model = Model(ModelConfig(model_path=h2_molecule_file))
    assert model.n_qubits == 4
    assert model.n_electrons == 2
    assert model.n_spins == 0  # multiplicity 1 -> 2S = 0
    assert model.ref_energy == pytest.approx(-1.137)


def test_openfermion_spin_from_multiplicity(tmp_path) -> None:
    """n_spins = multiplicity - 1 (triplet -> 2)."""
    from qmp.models.openfermion import Model, ModelConfig

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
