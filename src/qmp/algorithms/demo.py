"""Demo algorithm: print model info — a minimal algorithm for CLI testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qmp.algorithms._registry import action_class_dict, action_config_dict
from qmp.utility._build import SubConfigRef, build_model

logger = logging.getLogger(__name__)


@dataclass
class DemoConfig:
    """Configuration for the demo algorithm."""

    model: SubConfigRef | None = None
    network: SubConfigRef | None = None
    message: str = "Hello from qmp-kit demo!"


class Demo:
    """Prints model metadata."""

    def __init__(self, config: DemoConfig) -> None:
        self._config = config
        self._model = build_model(config.model)
        self._network = None
        if config.network is not None and self._model is not None:
            self._network = self._model.create_network(  # ty: ignore — model-owned
                config.network.name, config.network.params
            )

    def run(self) -> None:
        logger.info("Demo algorithm starting.")
        logger.info("Message: %s", self._config.message)
        if self._model is not None:
            logger.info("Model ref_energy: %s", self._model.ref_energy)
        else:
            logger.info("No model configured.")
        if self._network is not None:
            logger.info("Network available.")
        else:
            logger.info("No network configured.")
        logger.info("Demo algorithm finished.")


action_config_dict["demo"] = DemoConfig
action_class_dict["demo"] = Demo
