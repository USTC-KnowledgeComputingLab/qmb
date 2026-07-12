"""Tests for demo algorithm."""

from __future__ import annotations

from unittest import mock

from qmp.algorithms._registry import action_class_dict, action_config_dict
from qmp.algorithms.demo import Demo, DemoConfig
from qmp.models._build import SubConfigRef


def test_demo_config_defaults() -> None:
    cfg = DemoConfig()
    assert cfg.model is None
    assert cfg.network is None
    assert cfg.message is None
    assert cfg.sample_count == 1024


def test_demo_config_with_refs() -> None:
    cfg = DemoConfig(
        model=SubConfigRef(name="hubbard", params={"m": 2}),
        network=SubConfigRef(name="mlp/u1", params={"hidden_size": [64]}),
        message="test",
    )
    assert cfg.model.name == "hubbard"  # ty: ignore — None guard not needed in this test
    assert cfg.network.name == "mlp/u1"  # ty: ignore
    assert cfg.message == "test"


def test_demo_init_no_model_no_network() -> None:
    demo = Demo(DemoConfig())
    assert demo._model is None
    assert demo._network is None


def test_demo_init_with_model_only() -> None:
    with mock.patch("qmp.algorithms.demo.build_model", return_value=mock.Mock(ref_energy=0.0)):
        cfg = DemoConfig(model=SubConfigRef(name="hubbard"))
        demo = Demo(cfg)
        assert demo._model is not None
        assert demo._network is None


def test_demo_run_without_model() -> None:
    demo = Demo(DemoConfig())
    demo.run()


def test_demo_run_with_model_only() -> None:
    fake_model = mock.Mock(ref_energy=0.0)
    with mock.patch("qmp.algorithms.demo.build_model", return_value=fake_model):
        demo = Demo(DemoConfig(model=SubConfigRef(name="hubbard")))
        demo.run()


def test_demo_config_registration() -> None:
    assert action_config_dict["demo"] is DemoConfig
    assert action_class_dict["demo"] is Demo


# ---- end-to-end: model + network ----


def test_demo_init_builds_network_via_model() -> None:
    """With both refs, Demo builds the network through model.create_network."""
    cfg = DemoConfig(
        model=SubConfigRef(name="hubbard", params={"m": 2, "n": 1, "u": 4.0}),
        network=SubConfigRef(name="mlp/u1", params={"hidden_size": [16]}),
    )
    demo = Demo(cfg)
    assert demo._model is not None
    assert demo._network is not None


def test_demo_run_with_model_and_network() -> None:
    """The full run computes a finite <H> from a real model + network."""
    cfg = DemoConfig(
        model=SubConfigRef(name="hubbard", params={"m": 2, "n": 1, "u": 4.0}),
        network=SubConfigRef(name="mlp/u1", params={"hidden_size": [16]}),
        sample_count=32,
    )
    demo = Demo(cfg)
    demo.run()  # must not raise; exercises generate + apply_within_subspace


def test_demo_network_ref_without_model_stays_none() -> None:
    """A network ref without a model ref leaves the network unbuilt (guarded)."""
    cfg = DemoConfig(network=SubConfigRef(name="mlp/u1", params={"hidden_size": [16]}))
    demo = Demo(cfg)
    assert demo._model is None
    assert demo._network is None
    demo.run()  # must not raise
