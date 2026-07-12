"""Tests for algorithm registry."""

from __future__ import annotations

from qmp.algorithms._registry import ActionProto, action_class_dict, action_config_dict


class _FakeAction:
    def run(self, model: object) -> None:
        pass


def test_action_config_dict_registration() -> None:
    """Action config dict supports add/del."""
    action_config_dict["_test"] = type("_TestConfig", (), {})
    assert "_test" in action_config_dict
    del action_config_dict["_test"]
    assert "_test" not in action_config_dict


def test_action_class_dict_registration() -> None:
    """Action class dict supports add/del."""
    action_class_dict["_test"] = _FakeAction
    assert "_test" in action_class_dict
    del action_class_dict["_test"]
    assert "_test" not in action_class_dict


def test_action_proto_runtime_checkable() -> None:
    """ActionProto is runtime checkable via isinstance."""
    assert isinstance(_FakeAction(), ActionProto)


def test_action_proto_rejects_non_conforming() -> None:
    """Class without run() is not an ActionProto."""

    class _NoRun:
        pass

    assert not isinstance(_NoRun(), ActionProto)


def test_registry_isolation() -> None:
    """Previously deleted entries don't persist."""
    assert "_test" not in action_config_dict
    assert "_test" not in action_class_dict
