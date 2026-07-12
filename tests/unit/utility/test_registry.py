"""Tests for model registry registration."""

from __future__ import annotations

from qmp.models._model import model_config_dict, model_dict


def test_model_dict_is_dict() -> None:
    """model_dict is a mutable dict usable as a registry."""
    assert isinstance(model_dict, dict)


def test_model_config_dict_is_dict() -> None:
    """model_config_dict is a mutable dict."""
    assert isinstance(model_config_dict, dict)


def test_model_dict_registration() -> None:
    """Adding and removing from model_dict works."""
    model_dict["_test_entry"] = type("_TestModel", (), {})  # ty: ignore — test-only dummy
    assert model_dict["_test_entry"] is not None
    del model_dict["_test_entry"]
