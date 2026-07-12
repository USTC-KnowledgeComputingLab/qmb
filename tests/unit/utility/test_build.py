"""Tests for SubConfigRef and build_model."""

from __future__ import annotations

from dataclasses import dataclass
from unittest import mock

import dacite
import pytest

from qmp.models._build import SubConfigRef, build_model


@dataclass
class _FakeConfig:
    value: int = 1
    label: str = "default"


class _FakeImpl:
    def __init__(self, config: _FakeConfig) -> None:
        self.config = config


# ---- SubConfigRef dataclass ----


def test_sub_config_ref_defaults() -> None:
    ref = SubConfigRef(name="test")
    assert ref.name == "test"
    assert ref.params == {}


def test_sub_config_ref_with_params() -> None:
    ref = SubConfigRef(name="hubbard", params={"m": 2, "n": 2})
    assert ref.name == "hubbard"
    assert ref.params == {"m": 2, "n": 2}


# ---- build_model: None ref ----


def test_build_model_none() -> None:
    result = build_model(None)
    assert result is None


# ---- build_model: valid ref ----


def test_build_model_valid() -> None:
    with (
        mock.patch("qmp.models._build.model_config_dict", {"fake": _FakeConfig}),
        mock.patch("qmp.models._build.model_dict", {"fake": _FakeImpl}),
    ):
        ref = SubConfigRef(name="fake", params={"value": 42, "label": "test"})
        instance = build_model(ref)
        assert isinstance(instance, _FakeImpl)
        assert instance.config.value == 42
        assert instance.config.label == "test"


def test_build_model_default_params() -> None:
    with (
        mock.patch("qmp.models._build.model_config_dict", {"fake": _FakeConfig}),
        mock.patch("qmp.models._build.model_dict", {"fake": _FakeImpl}),
    ):
        ref = SubConfigRef(name="fake")
        instance = build_model(ref)
        assert instance.config.value == 1
        assert instance.config.label == "default"


# ---- build_model: error cases ----


def test_build_model_missing_name() -> None:
    with mock.patch("qmp.models._build.model_config_dict", {}), mock.patch("qmp.models._build.model_dict", {}):
        ref = SubConfigRef(name="nonexistent")
        with pytest.raises(KeyError, match="nonexistent"):
            build_model(ref)


def test_build_model_dacite_type_error() -> None:
    with (
        mock.patch("qmp.models._build.model_config_dict", {"fake": _FakeConfig}),
        mock.patch("qmp.models._build.model_dict", {"fake": _FakeImpl}),
    ):
        ref = SubConfigRef(name="fake", params={"value": "not_an_int"})
        with pytest.raises(dacite.DaciteError):
            build_model(ref)


def test_build_model_unknown_param_ignored() -> None:
    with (
        mock.patch("qmp.models._build.model_config_dict", {"fake": _FakeConfig}),
        mock.patch("qmp.models._build.model_dict", {"fake": _FakeImpl}),
    ):
        ref = SubConfigRef(name="fake", params={"value": 1, "extra": "ignored"})
        instance = build_model(ref)
        assert instance.config.value == 1


# ---- build_model: module lazy-import and integration ----


def test_build_model_config_without_impl_errors() -> None:
    """Name in config_dict but NOT in impl_dict → KeyError."""
    config_only = {"fake": _FakeConfig}
    empty_impl: dict[str, type[object]] = {}
    with (
        mock.patch("qmp.models._build.model_config_dict", config_only),
        mock.patch("qmp.models._build.model_dict", empty_impl),
    ):
        ref = SubConfigRef(name="fake")
        with pytest.raises(KeyError):
            build_model(ref)


def test_build_model_import_succeeds_but_no_registration() -> None:
    """Lazy import of a real module that doesn't register → KeyError."""
    with mock.patch("qmp.models._build.model_config_dict", {}), mock.patch("qmp.models._build.model_dict", {}):
        # "sys" is importable but not a model — _build will try importlib then check dict
        ref = SubConfigRef(name="sys")
        with pytest.raises(KeyError, match="sys"):
            build_model(ref)
