"""Demo algorithm: compute energy expectation <ψ|H|ψ>/<ψ|ψ> using a model and network."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from qmp.algorithms._registry import action_class_dict, action_config_dict
from qmp.models._build import SubConfigRef, build_model

logger = logging.getLogger(__name__)


@dataclass
class DemoConfig:
    """Configuration for the demo algorithm."""

    model: SubConfigRef | None = None
    network: SubConfigRef | None = None
    message: str | None = None
    sample_count: int = 1024


class Demo:
    """Compute and print the energy expectation <ψ|H|ψ>/<ψ|ψ>."""

    def __init__(self, config: DemoConfig) -> None:
        self._config = config
        self._model = build_model(config.model)
        self._network = None
        if config.network is not None and self._model is not None:
            self._network = self._model.create_network(config.network.name, config.network.params, rngs=nnx.Rngs(42))

    def run(self) -> None:
        logger.info("Demo algorithm starting.")
        if self._config.message:
            logger.info("Message: %s", self._config.message)

        if self._model is None:
            logger.info("No model configured — nothing to do.")
            return
        if self._network is None:
            logger.info("No network configured — nothing to do.")
            return

        key = jax.random.key(self._config.sample_count)
        configs, psi_complex, _ = self._network.generate(self._config.sample_count, key=key)
        logger.info("Sampled %d unique configs from network.", int(configs.shape[0]))

        psi_real = jnp.stack([psi_complex.real, psi_complex.imag], axis=1)
        h_psi = self._model.apply_within_subspace(configs, psi_real, configs)
        energy = jnp.sum(psi_real[:, 0] * h_psi[:, 0] + psi_real[:, 1] * h_psi[:, 1])
        norm = jnp.sum(psi_real[:, 0] ** 2 + psi_real[:, 1] ** 2)
        energy = energy / norm
        logger.info("<H> = %.10f", float(energy))

        if self._model.ref_energy != 0.0:
            logger.info("Ref  = %.10f", self._model.ref_energy)
            logger.info("Diff = %.10f", float(energy - self._model.ref_energy))
        logger.info("Demo algorithm finished.")


action_config_dict["demo"] = DemoConfig
action_class_dict["demo"] = Demo
