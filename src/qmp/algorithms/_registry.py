"""Algorithm registry.

``ActionProto`` defines the interface every algorithm must implement.
``action_dict`` is the global registry; algorithms register themselves via
``action_dict[name] = ConfigClass`` at module import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qmp.models._model import ModelProto

# re-export for convenience
from qmp.models._model import model_dict  # noqa: F401


@runtime_checkable
class ActionProto(Protocol):
    """Uniform interface for all algorithms."""

    def run(self, model: ModelProto[object]) -> None:
        """Execute the algorithm on the given model."""
        ...


action_dict: dict[str, type[object]] = {}
