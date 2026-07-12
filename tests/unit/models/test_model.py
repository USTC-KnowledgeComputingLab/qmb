"""Tests for the model protocol and registry."""

from __future__ import annotations

from qmp.models._model import ModelProto, model_dict


def test_model_dict_is_dict() -> None:
    """model_dict is a mutable dict usable as a registry."""
    assert isinstance(model_dict, dict)


def test_model_dict_registration() -> None:
    """Entries can be registered and retrieved by name."""

    class _Dummy(ModelProto[object]):
        ref_energy = 0.0
        network_dict: dict[str, object] = {}

    model_dict["_test_entry"] = _Dummy
    assert model_dict["_test_entry"] is _Dummy
    del model_dict["_test_entry"]
