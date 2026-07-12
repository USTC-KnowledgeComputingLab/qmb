"""Tests for TrimCI config, helpers, and registration."""

from __future__ import annotations

from qmp.algorithms.trim import Trim, TrimConfig


def test_trim_config_defaults() -> None:
    cfg = TrimConfig()
    assert cfg.max_rounds == 4
    assert cfg.pool_core_ratio == 10
    assert cfg.num_groups == 10
    assert cfg.local_keep_count == 64
    assert cfg.core_keep_count == 128
    assert cfg.first_cycle_keep_size == 10
    assert cfg.max_cycles == -1
    assert cfg.loss_name == "sum_filtered_angle_scaled_log"


def test_trim_registration() -> None:
    from qmp.algorithms._registry import action_class_dict, action_config_dict

    assert action_config_dict["trim"] is TrimConfig
    assert action_class_dict["trim"] is Trim
