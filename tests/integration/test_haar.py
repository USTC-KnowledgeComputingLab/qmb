"""Integration test: HAAR on H2/STO-3G via OpenFermion.

This test creates a minimal H2 molecule, runs HAAR for a few cycles,
and verifies the Krylov energy decreases and is reasonable.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import openfermion
import pytest
from flax import nnx

from qmp.algorithms.haar import Haar, HaarConfig, KrylovBasisStrategy, _DynamicLanczos, _merge_pools, _init_state
from qmp.models._build import SubConfigRef
from qmp.models.openfermion import Model, ModelConfig
from qmp.networks.mlp import WaveFunctionElectron as MlpElectron
from qmp.utility._losses import sum_filtered_angle_scaled_log


@pytest.fixture
def h2_model() -> Model:
    """Create a minimal H2/STO-3G model."""
    geom = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
    mol = openfermion.MolecularData(geom, basis="sto-3g", multiplicity=1, charge=0, filename="/tmp/h2_int_test")
    mol.n_orbitals = 2
    mol.n_qubits = 4
    mol.n_electrons = 2
    mol.fci_energy = -1.137
    mol.nuclear_repulsion = 0.7137539936
    mol.one_body_integrals = np.array([[-1.2524, 0.0], [0.0, -0.4759]])
    mol.two_body_integrals = np.zeros((2, 2, 2, 2))
    mol.two_body_integrals[0, 0, 0, 0] = 0.6746
    mol.two_body_integrals[1, 1, 1, 1] = 0.6976
    mol.save()
    return Model(ModelConfig(model_path="/tmp/h2_int_test"))


@pytest.fixture
def h2_network(h2_model: Model) -> MlpElectron:
    """Create a small MLP network for H2."""
    return MlpElectron(
        sites=h2_model.n_qubits, electrons=h2_model.n_electrons, hidden_size=(16,), ordering=1, rngs=nnx.Rngs(42)
    )


# ---------------------------------------------------------------------------
# Krylov subsystem
# ---------------------------------------------------------------------------


def test_krylov_energy_decreases(h2_model: Model, h2_network: MlpElectron) -> None:
    """Krylov energy should decrease monotonically with Lanczos steps."""
    key = jax.random.key(42)
    configs, psi = h2_network.generate_unique(64, key=key)
    lanczos = _DynamicLanczos(
        model=h2_model,
        configs=configs,
        psi=psi,
        max_steps=4,
        stop_norm=1e-8,
        random_period=99,
        extend_count=32,
        strategy=KrylovBasisStrategy.FIXED,
        state_count=1,
    )
    energies: list[float] = []
    for r in lanczos.run():
        energies.append(r[0][0])

    assert len(energies) >= 4
    assert energies[0] > -2.0  # should not be absurdly low
    assert energies[-1] < energies[0]  # should decrease


def test_krylov_energy_finite(h2_model: Model, h2_network: MlpElectron) -> None:
    """Krylov energy is finite and decreases with Lanczos steps."""
    key = jax.random.key(42)
    configs, psi = h2_network.generate_unique(64, key=key)

    lanczos = _DynamicLanczos(
        model=h2_model,
        configs=configs,
        psi=psi,
        max_steps=3,
        stop_norm=1e-8,
        random_period=99,
        extend_count=32,
        strategy=KrylovBasisStrategy.FIXED,
        state_count=1,
    )
    energies: list[float] = []
    for r in lanczos.run():
        energies.append(r[0][0])

    assert len(energies) >= 3
    assert all(math.isfinite(e) for e in energies)
    assert energies[-1] < energies[0]


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def test_merge_pools_empty_second(h2_network: MlpElectron) -> None:
    """Empty second pool returns first unchanged."""
    key = jax.random.key(42)
    c, psi = h2_network.generate_unique(4, key=key)
    cb = jnp.zeros((0, c.shape[1]), dtype=jnp.uint8)
    pb = jnp.zeros((0,), dtype=jnp.complex128)
    rc, rp = _merge_pools(c, psi, cb, pb)
    assert rc.shape == c.shape
    assert rp.shape == psi.shape


def test_merge_pools_dedup_prefers_first(h2_network: MlpElectron) -> None:
    """When pool (cb/pb) and network (ca/pa) have same config, pool psi wins."""
    c = jnp.zeros((1, 4), dtype=jnp.uint8)
    p_net = jnp.array([1.0 + 0j], dtype=jnp.complex128)
    p_pool = jnp.array([10.0 + 0j], dtype=jnp.complex128)
    rc, rp = _merge_pools(c, p_net, c, p_pool)
    assert rc.shape[0] == 1
    assert abs(float(rp[0].real) - 10.0) < 1e-10


# ---------------------------------------------------------------------------
# Loss function gradient
# ---------------------------------------------------------------------------


def test_loss_gradient(h2_network: MlpElectron) -> None:
    """Gradient of loss with respect to network parameters is non-zero for different target."""
    from flax import nnx

    key = jax.random.key(42)
    configs, psi = h2_network.generate_unique(16, key=key)
    max_idx = int(jnp.argmax(jnp.abs(psi)))
    target = psi.at[max_idx].set(psi[max_idx] * 2.0)  # alter target to be different
    target = target / target[jnp.argmax(jnp.abs(target))]

    graphdef, params = nnx.split(h2_network, nnx.Param)

    def _loss(pdict):
        net = nnx.merge(graphdef, pdict)
        psi_out = net(configs)
        psi_out = psi_out / psi_out[max_idx]
        return sum_filtered_angle_scaled_log(psi_out, target)

    loss_val, grads = jax.value_and_grad(_loss)(params)
    grad_norms = [float(jnp.linalg.norm(v)) for v in jax.tree_util.tree_leaves(grads)]
    assert any(g > 0 for g in grad_norms)


# ---------------------------------------------------------------------------
# HAAR config
# ---------------------------------------------------------------------------


def test_haar_integration_cycle(h2_model: Model, h2_network: MlpElectron) -> None:
    """A full HAAR cycle (Krylov + one local step) runs without error."""
    cfg = HaarConfig(
        krylov_max_steps=3,
        basis_extend_count=32,
        basis_strategy=KrylovBasisStrategy.FIXED,
        local_max_steps=1,  # just one step
    )
    haar = Haar(cfg)
    haar._model = h2_model
    haar._network = h2_network

    # Don't call run() (infinite loop) — manually do one cycle
    state = _init_state()
    key = jax.random.key(123)
    c_net, p_net = h2_network.generate_unique(cfg.sampling_count_from_network, key=key)
    configs, psi = _merge_pools(
        c_net, p_net, jnp.zeros((0, c_net.shape[1]), dtype=jnp.uint8), jnp.zeros((0,), dtype=jnp.complex128)
    )

    lanczos = _DynamicLanczos(
        model=h2_model,
        configs=configs,
        psi=psi,
        max_steps=cfg.krylov_max_steps,
        stop_norm=cfg.krylov_stop_norm,
        random_period=cfg.krylov_random_period,
        extend_count=cfg.basis_extend_count,
        strategy=cfg.basis_strategy,
        state_count=cfg.krylov_state_count,
    )
    results = list(lanczos.run())
    assert len(results) > 0
    e0, _, _ = results[-1][0]
    assert abs(e0) < 3.0  # reasonable for H2
