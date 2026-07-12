"""Shared pytest configuration.

The Fermi Hamiltonian subsystem uses float64 coefficients and complex128
amplitudes throughout. The CUDA FFI kernels declare F64 buffers, so x64 must be
enabled for JAX; otherwise ``jnp`` silently truncates float64 to float32 and the
FFI operand dtypes mismatch. Enabling it globally also makes the pure-JAX
fallback operate at the intended precision.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
