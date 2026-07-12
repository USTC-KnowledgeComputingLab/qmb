"""Tests for the FCIDUMP model."""

from __future__ import annotations

import gzip
import pathlib
import pickle

import jax.numpy as jnp
import pytest

import qmp.models.fcidump as fcidump_module
from qmp.models._model import model_dict
from qmp.models.fcidump import Model, ModelConfig, read_fcidump

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
