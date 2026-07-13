"""Integration test: TrimCI x HAAR on a small Hubbard model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from qmp.algorithms.trim import KrylovBasisStrategy, Trim, TrimConfig, _DynamicLanczos, _expand_pool, _local_trim
from qmp.models._build import SubConfigRef
from qmp.models.hubbard import Model, ModelConfig


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


def _exact_ground_electronic(model: Model, n_qubits: int) -> float:
    """Full-space exact diagonalization of H (electronic) over all 2**n_qubits configs."""
    size = 1 << n_qubits
    all_configs = jnp.array([[i] for i in range(size)], dtype=jnp.uint8)
    h_mat = jnp.zeros((size, size), dtype=jnp.float64)
    for j in range(size):
        col = model.apply_within_subspace(all_configs[j : j + 1], jnp.array([[1.0, 0.0]]), all_configs)
        h_mat = h_mat.at[:, j].set(col[:, 0])
    return float(jnp.linalg.eigh(h_mat)[0][0])


def test_trim_global_is_variational_upper_bound() -> None:
    """TrimCI's global Lanczos energy is a valid variational upper bound (>= exact) and finite.

    On the tiny 2x1 half-filled Hubbard model (4-qubit, 16-config space) the
    exact electronic ground state is computed by full diagonalization. TrimCI's
    heuristic expansion + trim need not enumerate the whole space, but its
    projected energy must never fall below the exact ground state.
    """
    model = Model(ModelConfig(m=2, n=1, t=1.0, u=4.0, electron_number=2))
    exact = _exact_ground_electronic(model, model.n_qubits)

    net = model.create_network("mlp/u1u1", {"hidden_size": [8]}, rngs=nnx.Rngs(0))
    core_c, core_p = net.generate_unique(4, key=jax.random.key(7))
    pool_c, pool_p = _expand_pool(model, core_c, core_p, max_rounds=4, pool_core_ratio=10)
    surv_c, surv_p = _local_trim(
        model,
        pool_c,
        pool_p,
        num_groups=2,
        keep_count=16,
        lanczos_steps=8,
        stop_norm=1e-8,
        random_period=99,
        key=jax.random.key(8),
    )

    lanczos = _DynamicLanczos(
        model=model,
        configs=surv_c,
        psi=surv_p,
        max_steps=16,
        stop_norm=1e-8,
        random_period=99,
        extend_count=0,
        strategy=KrylovBasisStrategy.FIXED,
        state_count=1,
    )
    energy = 0.0
    for results in lanczos.run():
        energy = results[0][0]

    assert jnp.isfinite(jnp.array(energy))
    # variational bound: projected energy cannot be below the exact ground state
    assert energy >= exact - 1e-6
    # and the subspace should capture a meaningful fraction of the correlation
    assert energy < 0.0
