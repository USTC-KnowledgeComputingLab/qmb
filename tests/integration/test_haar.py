"""Integration test: HAAR on H2/STO-3G via OpenFermion.

Uses PySCF to generate real H2 integrals when available,
otherwise falls back to pre-computed values.

Note: ``model.compute_diagonal_within_subspace`` and Krylov energies
are *electronic* only (exclude nuclear repulsion).  ``model.ref_energy``
is the full FCI total energy including nuclear repulsion.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qmp.algorithms.haar import Haar, HaarConfig, KrylovBasisStrategy, _DynamicLanczos, _merge_pools
from qmp.models._build import SubConfigRef
from qmp.models.openfermion import Model, ModelConfig
from qmp.networks.mlp import WaveFunctionElectron as MlpElectron
from qmp.utility._losses import sum_filtered_angle_scaled_log

# pre-computed H2/STO-3G values (electronic part = FCI - nuclear_repulsion)
_H2_FCI_TOTAL = -1.1372701747
_H2_NUCLEAR = 0.7137539937
_H2_ELECTRONIC = _H2_FCI_TOTAL - _H2_NUCLEAR


def _make_h2_molecule(path: str) -> None:
    """Generate real H2/STO-3G data, falling back to pre-computed if PySCF unavailable."""
    try:
        from openfermion.chem import MolecularData
        from openfermionpyscf import run_pyscf

        geom = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
        mol = MolecularData(geom, "sto-3g", multiplicity=1, charge=0, filename=path)
        run_pyscf(mol, run_fci=True)
    except ImportError:
        import numpy as np
        import openfermion

        geom = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
        mol = openfermion.MolecularData(geom, basis="sto-3g", multiplicity=1, charge=0, filename=path)
        mol.n_orbitals = 2
        mol.n_qubits = 4
        mol.n_electrons = 2
        mol.fci_energy = _H2_FCI_TOTAL
        mol.nuclear_repulsion = _H2_NUCLEAR
        mol.one_body_integrals = np.array([[-1.252463573, -0.0], [-0.0, -0.475948715]])
        mol.two_body_integrals = np.zeros((2, 2, 2, 2))
        mol.two_body_integrals[0, 0, 0, 0] = 0.674493166
        mol.two_body_integrals[0, 0, 1, 1] = 0.181287518
        mol.two_body_integrals[0, 1, 0, 1] = 0.663472101
        mol.two_body_integrals[0, 1, 1, 0] = 0.697397917
        mol.two_body_integrals[1, 0, 0, 1] = 0.663472101
        mol.two_body_integrals[1, 0, 1, 0] = 0.181287518
        mol.two_body_integrals[1, 1, 1, 1] = 0.697397917
        mol.save()


@pytest.fixture(scope="module")
def h2_model() -> Model:
    """Create H2/STO-3G model from real PySCF data (or pre-computed fallback)."""
    _make_h2_molecule("/tmp/h2_int_test")
    return Model(ModelConfig(model_path="/tmp/h2_int_test"))


@pytest.fixture(scope="module")
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
    assert energies[0] > -2.0
    assert energies[-1] < energies[0]
    # Krylov should converge to the electronic energy (no nuclear repulsion)
    assert energies[-1] >= _H2_ELECTRONIC - 1e-6


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
    target = psi.at[max_idx].set(psi[max_idx] * 2.0)
    target = target / target[jnp.argmax(jnp.abs(target))]

    graphdef, params = nnx.split(h2_network, nnx.Param)

    def _loss(pdict):
        net = nnx.merge(graphdef, pdict)
        psi_out = net(configs)
        psi_out = psi_out / psi_out[max_idx]
        return sum_filtered_angle_scaled_log(psi_out, target)

    _loss_val, grads = jax.value_and_grad(_loss)(params)
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
        local_max_steps=1,
        max_cycles=1,
    )
    haar = Haar(cfg)
    haar._model = h2_model
    haar._network = h2_network
    haar.run()
