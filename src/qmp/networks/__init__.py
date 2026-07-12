"""Neural network wave-function ansatzes (autoregressive neural quantum states)."""

from __future__ import annotations

from . import mlp
from ._protocol import NetworkProto

__all__ = ["NetworkProto", "mlp"]
