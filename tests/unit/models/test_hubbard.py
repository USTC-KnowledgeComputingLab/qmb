"""Tests for the Hubbard model."""

from __future__ import annotations

import jax.numpy as jnp

from qmp.models._model import model_dict
from qmp.models.hubbard import Model, ModelConfig


def test_hubbard_registered() -> None:
    """Hubbard model registers itself in model_dict."""
    assert model_dict["hubbard"] is Model


def test_hubbard_n_qubits() -> None:
    """n_qubits = m * n * 2."""
    model = Model(ModelConfig(m=2, n=1))
    assert model.n_qubits == 4


def test_hubbard_default_half_filling() -> None:
    """electron_number defaults to m * n (half filling)."""
    model = Model(ModelConfig(m=2, n=2))
    assert model.electron_number == 4


def test_hubbard_hamiltonian_terms() -> None:
    """A 2x1 lattice with t=1, u=2, mu=0 produces the expected term set."""
    terms = Model._prepare_hamiltonian(ModelConfig(m=2, n=1, t=1.0, u=2.0, mu=0.0))
    # hopping: 2 spins x 2 directions = 4 terms, coef -1
    # on-site: 2 sites x 1 term = 2 terms, coef 2
    # mu = 0 so chemical potential terms have coef 0 (still present)
    assert terms[((0, 1), (2, 0))] == -1.0  # up hop site1->site0
    assert terms[((2, 1), (0, 0))] == -1.0  # up hop site0->site1
    assert terms[((0, 1), (0, 0), (1, 1), (1, 0))] == 2.0  # on-site site (0,0)


def test_hubbard_ref_energy() -> None:
    """ref_energy is passed through from config."""
    model = Model(ModelConfig(m=1, n=1, ref_energy=-3.5))
    assert model.ref_energy == -3.5


def test_hubbard_show_config() -> None:
    """show_config renders occupation as up/down arrows."""
    model = Model(ModelConfig(m=2, n=1))
    # site0 spin-up occupied (bit0=1) -> config byte = 0b00000001
    config = jnp.array([0b00000001], dtype=jnp.uint8)
    rendered = model.show_config(config)
    assert isinstance(rendered, str)
    assert "↑" in rendered


def test_hubbard_network_dict_empty() -> None:
    """network_dict is empty this round."""
    assert Model.network_dict == {}
