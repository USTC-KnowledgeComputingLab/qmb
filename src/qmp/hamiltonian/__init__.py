"""
Hamiltonian module for quantum many-body systems.

This module provides the Hamiltonian class for storing and processing
Hamiltonian terms using bit-operation-optimized CUDA kernels via JAX FFI.
"""

from __future__ import annotations

from ._hamiltonian import Hamiltonian

__all__ = ["Hamiltonian"]
