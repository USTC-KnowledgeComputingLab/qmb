"""Model protocol and registry.

``ModelProto`` defines the uniform interface every model exposes. ``model_dict``
is the plugin registry mapping model names to their classes; models register
themselves at import time via ``model_dict[name] = Model``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    import jax

config_t = TypeVar("config_t")


class ModelProto(Protocol[config_t]):
    """Uniform interface implemented by all models."""

    ref_energy: float
    network_dict: dict[str, object]

    def __init__(self, config: config_t) -> None:
        """Build the model from its configuration."""

    def compute_diagonal_within_subspace(self, configs: jax.Array) -> jax.Array:
        """Compute diagonal Hamiltonian elements for each configuration."""

    def apply_within_subspace(
        self,
        configs_i: jax.Array,
        psi_i: jax.Array,
        configs_j: jax.Array,
        *,
        direction: int = 0,
    ) -> jax.Array:
        """Apply H|psi_i> projected onto the configs_j subspace."""

    def find_all_relative_configs(
        self,
        configs_i: jax.Array,
        psi_i: jax.Array,
        configs_exclude: jax.Array | None = None,
        *,
        hash_capacity: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Enumerate all unique new configurations reachable via H."""

    def find_topk_relative_configs(
        self,
        configs_i: jax.Array,
        psi_i: jax.Array,
        count_selected: int,
        configs_exclude: jax.Array | None = None,
    ) -> jax.Array:
        """Select the top-K most important new configurations."""

    def show_config(self, config: jax.Array) -> str:
        """Render a bit-packed configuration as a human-readable string."""


model_dict: dict[str, type[ModelProto[object]]] = {}
