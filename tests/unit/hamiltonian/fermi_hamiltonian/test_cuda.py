"""CUDA regression tests (requires GPU and compiled .so)."""

from __future__ import annotations

import jax
import pytest

cuda_available = any("cuda" in str(d).lower() for d in jax.devices())
pytestmark = pytest.mark.skipif(not cuda_available, reason="No CUDA device available")
