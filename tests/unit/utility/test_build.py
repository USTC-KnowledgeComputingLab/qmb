"""Tests for SubConfigRef and build_from_ref."""

from __future__ import annotations

from dataclasses import dataclass

import dacite
import pytest

from qmp.utility._build import SubConfigRef, build_from_ref


@dataclass
class _FakeConfig:
    value: int = 1
    label: str = "default"


class _FakeImpl:
    def __init__(self, config: _FakeConfig) -> None:
        self.config = config


FAKE_CONFIG_DICT: dict[str, type[object]] = {"fake": _FakeConfig}
FAKE_IMPL_DICT: dict[str, type[object]] = {"fake": _FakeImpl}


# ---- SubConfigRef dataclass ----


def test_sub_config_ref_defaults() -> None:
    ref = SubConfigRef(name="test")
    assert ref.name == "test"
    assert ref.params == {}


def test_sub_config_ref_with_params() -> None:
    ref = SubConfigRef(name="hubbard", params={"m": 2, "n": 2})
    assert ref.name == "hubbard"
    assert ref.params == {"m": 2, "n": 2}


# ---- build_from_ref: None ref ----


def test_build_from_ref_none() -> None:
    result = build_from_ref(None, "models", config_dict=FAKE_CONFIG_DICT, impl_dict=FAKE_IMPL_DICT)
    assert result is None


# ---- build_from_ref: valid ref ----


def test_build_from_ref_valid() -> None:
    ref = SubConfigRef(name="fake", params={"value": 42, "label": "test"})
    instance = build_from_ref(ref, "models", config_dict=FAKE_CONFIG_DICT, impl_dict=FAKE_IMPL_DICT)
    assert isinstance(instance, _FakeImpl)
    assert instance.config.value == 42
    assert instance.config.label == "test"


def test_build_from_ref_default_params() -> None:
    """Empty params dict → config gets dataclass defaults."""
    ref = SubConfigRef(name="fake")
    instance = build_from_ref(ref, "models", config_dict=FAKE_CONFIG_DICT, impl_dict=FAKE_IMPL_DICT)
    assert instance.config.value == 1
    assert instance.config.label == "default"


# ---- build_from_ref: error cases ----


def test_build_from_ref_missing_name() -> None:
    """Name not in config_dict → ModuleNotFoundError caught, KeyError raised."""
    ref = SubConfigRef(name="nonexistent")
    with pytest.raises(KeyError, match="nonexistent"):
        build_from_ref(ref, "models", config_dict=FAKE_CONFIG_DICT, impl_dict=FAKE_IMPL_DICT)


def test_build_from_ref_dacite_type_error() -> None:
    """Wrong param type → dacite raises."""
    ref = SubConfigRef(name="fake", params={"value": "not_an_int"})
    with pytest.raises(dacite.DaciteError):
        build_from_ref(ref, "models", config_dict=FAKE_CONFIG_DICT, impl_dict=FAKE_IMPL_DICT)


def test_build_from_ref_unknown_param_ignored() -> None:
    """Extra param not in config dataclass → dacite silently ignores by default."""
    ref = SubConfigRef(name="fake", params={"value": 1, "extra": "ignored"})
    instance = build_from_ref(ref, "models", config_dict=FAKE_CONFIG_DICT, impl_dict=FAKE_IMPL_DICT)
    assert instance.config.value == 1
