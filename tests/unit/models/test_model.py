"""Tests for the model protocol and registry."""

from __future__ import annotations

import importlib
from typing import ClassVar

import pytest

from qmp.models._model import ModelProto, model_config_dict, model_dict

_MODEL_NAMES = ("fcidump", "hubbard", "openfermion")
_REQUIRED_METHODS = (
    "compute_diagonal_within_subspace",
    "apply_within_subspace",
    "find_all_relative_configs",
    "find_topk_relative_configs",
    "show_config",
    "create_network",
)


@pytest.fixture(autouse=True)
def _register_models() -> None:
    """Import the model modules so they self-register into model_dict.

    Mirrors how the CLI triggers registration via importlib.import_module.
    """
    for name in _MODEL_NAMES:
        importlib.import_module(f"qmp.models.{name}")


def test_model_dict_is_dict() -> None:
    """model_dict is a mutable dict usable as a registry."""
    assert isinstance(model_dict, dict)


def test_model_config_dict_is_dict() -> None:
    """model_config_dict is a mutable dict usable as a registry."""
    assert isinstance(model_config_dict, dict)


def test_model_dict_registration() -> None:
    """Entries can be registered and retrieved by name."""

    class _Dummy(ModelProto[object]):
        ref_energy = 0.0
        network_dict: ClassVar[dict[str, object]] = {}

    model_dict["_test_entry"] = _Dummy
    assert model_dict["_test_entry"] is _Dummy
    del model_dict["_test_entry"]


def test_expected_models_registered() -> None:
    """The three migrated models are all present in the registry."""
    assert set(_MODEL_NAMES) <= set(model_dict)


@pytest.mark.parametrize("name", _MODEL_NAMES)
def test_registered_model_implements_protocol(name: str) -> None:
    """Every registered model class exposes the full ModelProto operator surface."""
    model_class = model_dict[name]
    for method in _REQUIRED_METHODS:
        assert callable(getattr(model_class, method, None)), f"{name} missing {method}"
    assert isinstance(model_class.network_dict, dict)
