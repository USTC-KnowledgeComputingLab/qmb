"""Tests for the FCIDUMP model."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import qmp.models.fcidump as fcidump_module
from qmp.models._model import model_dict
from qmp.models.fcidump import Model, ModelConfig

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
