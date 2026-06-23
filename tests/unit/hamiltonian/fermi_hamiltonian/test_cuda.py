"""CUDA regression tests (requires GPU and compiled .so)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

cuda_available = any("cuda" in str(d).lower() for d in jax.devices())

from qmp.hamiltonian.fermi_hamiltonian._hamiltonian import FermiHamiltonian
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_jax import (
    compute_diagonal_within_subspace as jax_diag,
)
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_prepare import prepare

pytestmark = pytest.mark.skipif(not cuda_available, reason="No CUDA device available")
