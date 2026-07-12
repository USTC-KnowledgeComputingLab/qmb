"""
The qmp package provides tools and algorithms for solving quantum many-body systems.

For more details, please refer to the README.md and AGENTS.md files.
"""

from __future__ import annotations

import jax

from ._version import __version__, version

# Enable 64-bit precision globally: log-amplitudes, phases and Hamiltonian
# coefficients are accumulated in float64 across the package.
jax.config.update("jax_enable_x64", True)

__all__ = ["__version__", "version"]
