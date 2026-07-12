"""Integration test: TrimCI x HAAR on a small Hubbard model."""

from __future__ import annotations

import jax.numpy as jnp

from qmp.algorithms.trim import Trim, TrimConfig
from qmp.models._build import SubConfigRef


def test_trim_runs_one_cycle() -> None:
    cfg = TrimConfig(
        model=SubConfigRef(name="hubbard", params={"m": 2, "n": 1, "t": 1.0, "u": 4.0, "electron_number": 2}),
        network=SubConfigRef(name="mlp/u1u1", params={"hidden_size": [8]}),
        sampling_count_from_network=8,
        sampling_count_from_pool=8,
        max_rounds=2,
        pool_core_ratio=3,
        num_groups=2,
        local_keep_count=4,
        local_lanczos_steps=4,
        global_lanczos_steps=6,
        core_keep_count=8,
        first_cycle_keep_size=4,
        local_max_steps=20,
        max_cycles=1,
        checkpoint_path=None,
    )
    trim = Trim(cfg)
    trim.run()

    state = trim._state
    assert state["trim"]["global"] == 1
    assert state["trim"]["pool"] is not None
    configs, psi, _counts = state["trim"]["pool"]
    assert configs.shape[0] == psi.shape[0]
    assert configs.shape[0] > 0
    excited = state["trim"]["excited"]
    assert len(excited) >= 1
    e0 = excited[0][0]
    assert jnp.isfinite(jnp.array(e0))
    assert -10.0 < float(e0) < 5.0


def test_trim_target_normalized() -> None:
    cfg = TrimConfig(
        model=SubConfigRef(name="hubbard", params={"m": 2, "n": 1, "t": 1.0, "u": 4.0, "electron_number": 2}),
        network=SubConfigRef(name="mlp/u1u1", params={"hidden_size": [8]}),
        sampling_count_from_network=8,
        sampling_count_from_pool=8,
        max_rounds=1,
        pool_core_ratio=2,
        num_groups=2,
        local_keep_count=4,
        local_lanczos_steps=4,
        global_lanczos_steps=6,
        local_max_steps=10,
        max_cycles=1,
    )
    trim = Trim(cfg)
    trim.run()
    for e, cfg_i, p in trim._state["trim"]["excited"]:
        assert bool(jnp.isfinite(jnp.array(e)))
        assert cfg_i.shape[0] == p.shape[0]
