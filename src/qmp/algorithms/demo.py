"""Demo algorithm: print model info — a minimal algorithm for CLI testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qmp.algorithms._registry import action_class_dict, action_config_dict
from qmp.models._model import model_config_dict, model_dict
from qmp.networks._registry import network_class_dict, network_config_dict
from qmp.utility._build import SubConfigRef, build_from_ref

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
        self._model = build_from_ref(config.model, "models", config_dict=model_config_dict, impl_dict=model_dict)
        self._network = build_from_ref(
            config.network, "networks", config_dict=network_config_dict, impl_dict=network_class_dict
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
