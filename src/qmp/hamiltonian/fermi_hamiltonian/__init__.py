"""
Fermi Hamiltonian module.

Provides FermiHamiltonian for fermionic quantum many-body systems
using bit-operation-optimized CUDA kernels via JAX FFI.
"""

from __future__ import annotations

from ._hamiltonian import FermiHamiltonian

__all__ = ["FermiHamiltonian"]
