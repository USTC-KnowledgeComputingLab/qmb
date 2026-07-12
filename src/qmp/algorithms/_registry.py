"""Algorithm registry.

``action_config_dict`` maps action names to their config dataclass type.
``action_class_dict`` maps action names to their implementation class.
Algorithms register themselves at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qmp.models._model import ModelProto


@runtime_checkable
class ActionProto(Protocol):
    """Uniform interface for all algorithms."""

    def run(self, model: ModelProto[object]) -> None:
        """Execute the algorithm."""
        ...


action_config_dict: dict[str, type[object]] = {}
action_class_dict: dict[str, type[object]] = {}
