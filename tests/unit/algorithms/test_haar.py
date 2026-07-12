"""Tests for HAAR config and registration."""

from __future__ import annotations

from qmp.algorithms.haar import Haar, HaarConfig, KrylovBasisStrategy


def test_haar_config_defaults() -> None:
    cfg = HaarConfig()
    assert cfg.basis_strategy == KrylovBasisStrategy.ADAPTIVE
    assert cfg.krylov_max_steps == 32
    assert cfg.krylov_random_period == 31
    assert cfg.local_max_steps == 10000


def test_krylov_strategy_enum_values() -> None:
    assert KrylovBasisStrategy.FIXED.value == "fixed"
    assert KrylovBasisStrategy.ADAPTIVE.value == "adaptive"


def test_haar_config_registration() -> None:
    from qmp.algorithms._registry import action_class_dict, action_config_dict

    assert action_config_dict["haar"] is HaarConfig
    assert action_class_dict["haar"] is Haar
