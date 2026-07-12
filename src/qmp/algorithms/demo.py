"""Demo algorithm: print model info — a minimal algorithm for CLI testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qmp.algorithms._registry import action_dict
from qmp.models._model import ModelProto

logger = logging.getLogger(__name__)


@dataclass
class DemoConfig:
    """Configuration for the demo algorithm."""

    message: str = "Hello from qmp-kit demo!"
    show_config_count: int = 0


class Demo:
    """Prints model metadata."""

    def __init__(self, config: DemoConfig) -> None:
        self._config = config

    def run(self, model: ModelProto[object]) -> None:
        logger.info("Demo algorithm starting.")
        logger.info("Message: %s", self._config.message)
        logger.info("Model ref_energy: %s", model.ref_energy)
        logger.info("Model network_dict keys: %s", list(model.network_dict.keys()))
        logger.info("Demo algorithm finished.")


action_dict["demo"] = DemoConfig
