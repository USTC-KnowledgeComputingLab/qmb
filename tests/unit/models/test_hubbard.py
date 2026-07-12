"""Tests for the Hubbard model."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

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


def test_hubbard_explicit_electron_number() -> None:
    """An explicit electron_number is preserved (not overwritten by half filling)."""
    model = Model(ModelConfig(m=2, n=2, electron_number=3))
    assert model.electron_number == 3


def test_hubbard_devices_default() -> None:
    """devices defaults to a single local CPU entry."""
    assert ModelConfig(m=1, n=1).devices == ["localhost:cpu:0"]


def test_hubbard_invalid_dimensions() -> None:
    """Non-positive lattice dimensions raise ValueError."""
    with pytest.raises(ValueError, match="positive integers"):
        ModelConfig(m=0, n=1)


def test_hubbard_electron_number_out_of_bounds_high() -> None:
    """electron_number above 2*m*n raises ValueError."""
    with pytest.raises(ValueError, match="out of bounds"):
        ModelConfig(m=2, n=2, electron_number=99)


def test_hubbard_electron_number_out_of_bounds_negative() -> None:
    """Negative electron_number raises ValueError."""
    with pytest.raises(ValueError, match="out of bounds"):
        ModelConfig(m=2, n=2, electron_number=-1)


def test_hubbard_hamiltonian_terms() -> None:
    """A 2x1 lattice with t=1, u=2, mu=0 produces the expected term set."""
    terms = Model._prepare_hamiltonian(ModelConfig(m=2, n=1, t=1.0, u=2.0, mu=0.0))
    # hopping: 2 spins x 2 directions = 4 terms, coef -1
    # on-site: 2 sites x 1 term = 2 terms, coef 2
    # mu = 0 so chemical potential terms have coef 0 (still present)
    assert terms[((0, 1), (2, 0))] == -1.0  # up hop site1->site0
    assert terms[((2, 1), (0, 0))] == -1.0  # up hop site0->site1
    assert terms[((0, 1), (0, 0), (1, 1), (1, 0))] == 2.0  # on-site site (0,0)


def test_hubbard_vertical_hopping() -> None:
    """A 1x2 lattice produces vertical-bond hopping between the two rows."""
    terms = Model._prepare_hamiltonian(ModelConfig(m=1, n=2, t=1.0, u=0.0, mu=0.0))
    # site (0,0) up = qubit 0, site (0,1) up = qubit 2; down = qubits 1, 3
    assert terms[((2, 1), (0, 0))] == -1.0  # up hop row0->row1
    assert terms[((0, 1), (2, 0))] == -1.0  # up hop row1->row0
    assert terms[((3, 1), (1, 0))] == -1.0  # down hop row0->row1
    assert terms[((1, 1), (3, 0))] == -1.0  # down hop row1->row0


def test_hubbard_chemical_potential_coefficient() -> None:
    """Chemical potential terms carry coefficient -mu on each spin-orbital."""
    terms = Model._prepare_hamiltonian(ModelConfig(m=1, n=1, t=1.0, u=0.0, mu=0.7))
    assert terms[((0, 1), (0, 0))] == pytest.approx(-0.7)  # spin up number
    assert terms[((1, 1), (1, 0))] == pytest.approx(-0.7)  # spin down number


def test_hubbard_2x2_term_counts() -> None:
    """A 2x2 lattice has 4 bonds -> 16 directed spin hopping terms and 4 on-site terms."""
    terms = Model._prepare_hamiltonian(ModelConfig(m=2, n=2, t=1.0, u=4.0, mu=0.0))
    hopping = {key: value for key, value in terms.items() if len(key) == 2 and value == -1.0}
    on_site = {key: value for key, value in terms.items() if len(key) == 4}
    # 4 bonds (2 horizontal + 2 vertical) x 2 spins x 2 directions = 16
    assert len(hopping) == 16
    # one on-site interaction per site
    assert len(on_site) == 4
    assert set(on_site.values()) == {4.0}


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


def test_hubbard_show_config_all_occupations() -> None:
    """show_config renders empty, up, down, and doubly-occupied sites over rows."""
    model = Model(ModelConfig(m=1, n=2))
    # row0 = site0 doubly occupied (bits 0,1); row1 = site1 empty
    assert model.show_config(jnp.array([0b00000011], dtype=jnp.uint8)) == "[↕. ]"
    # row0 = site0 up (bit0); row1 = site1 down (bit3)
    assert model.show_config(jnp.array([0b00001001], dtype=jnp.uint8)) == "[↑.↓]"
    # all empty
    assert model.show_config(jnp.array([0b00000000], dtype=jnp.uint8)) == "[ . ]"


def test_hubbard_network_dict_empty() -> None:
    """network_dict is empty this round."""
    assert Model.network_dict == {}


def test_hubbard_diagonal_on_site_interaction() -> None:
    """Diagonal element of a single site equals U only when doubly occupied (U=3, mu=0)."""
    model = Model(ModelConfig(m=1, n=1, u=3.0, mu=0.0))
    configs = jnp.array([[0b00], [0b01], [0b10], [0b11]], dtype=jnp.uint8)
    diagonal = model.compute_diagonal_within_subspace(configs)
    assert [float(value) for value in diagonal[:, 0]] == [0.0, 0.0, 0.0, 3.0]


def test_hubbard_diagonal_chemical_potential() -> None:
    """Diagonal element counts occupied orbitals times -mu (mu=0.5, U=0)."""
    model = Model(ModelConfig(m=1, n=1, u=0.0, mu=0.5))
    configs = jnp.array([[0b00], [0b01], [0b10], [0b11]], dtype=jnp.uint8)
    diagonal = model.compute_diagonal_within_subspace(configs)
    assert [float(value) for value in diagonal[:, 0]] == [0.0, -0.5, -0.5, -1.0]


def test_hubbard_apply_within_hopping_amplitude() -> None:
    """Hopping moves an up electron site0->site1 with amplitude -t (t=1)."""
    model = Model(ModelConfig(m=2, n=1, t=1.0, u=0.0, mu=0.0))
    configs_i = jnp.array([[0b0001]], dtype=jnp.uint8)  # site0 spin-up (qubit 0)
    configs_j = jnp.array([[0b0100]], dtype=jnp.uint8)  # site1 spin-up (qubit 2)
    psi_i = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    result = model.apply_within_subspace(configs_i, psi_i, configs_j)
    assert float(result[0, 0]) == pytest.approx(-1.0)
    assert float(result[0, 1]) == pytest.approx(0.0)


def test_hubbard_find_all_relative_configs_forwarding() -> None:
    """find_all_relative_configs forwards with exclude=None and returns padded arrays."""
    model = Model(ModelConfig(m=2, n=1, t=1.0, u=0.0, mu=0.0))
    configs_i = jnp.array([[0b0001]], dtype=jnp.uint8)
    psi_i = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    keys, values, count = model.find_all_relative_configs(configs_i, psi_i, hash_capacity=16)
    assert keys.shape == (16, 1)
    assert values.shape == (16, 2)
    assert int(count) == 2  # hopping to site1 for both directions from the single occupied config


def test_hubbard_find_topk_relative_configs_forwarding() -> None:
    """find_topk_relative_configs forwards with exclude=None and returns count_selected rows."""
    model = Model(ModelConfig(m=2, n=1, t=1.0, u=0.0, mu=0.0))
    configs_i = jnp.array([[0b0001]], dtype=jnp.uint8)
    psi_i = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    selected = model.find_topk_relative_configs(configs_i, psi_i, 2)
    assert selected.shape == (2, 1)
