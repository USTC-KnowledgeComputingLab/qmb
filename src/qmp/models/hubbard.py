"""Hubbard model on a two-dimensional lattice."""

from __future__ import annotations

import dataclasses
import logging
from typing import ClassVar

import jax
import jax.numpy as jnp

from qmp.hamiltonian.fermi_hamiltonian import FermiHamiltonian

from ._model import ModelProto, model_dict

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelConfig:
    """Configuration for the Hubbard model."""

    m: int
    n: int
    t: float = 1.0
    u: float = 0.0
    mu: float = 0.0
    electron_number: int | None = None
    ref_energy: float = 0.0
    devices: list[str] = dataclasses.field(default_factory=lambda: ["localhost:cpu:0"])

    def __post_init__(self) -> None:
        if self.electron_number is None:
            self.electron_number = self.m * self.n
        if self.m <= 0 or self.n <= 0:
            raise ValueError("The dimensions of the Hubbard model must be positive integers.")
        if self.electron_number < 0 or self.electron_number > 2 * self.m * self.n:
            raise ValueError(
                f"The electron number {self.electron_number} is out of bounds for a {self.m}x{self.n} lattice."
            )


class Model(ModelProto[ModelConfig]):
    """Hubbard model wrapping a FermiHamiltonian."""

    network_dict: ClassVar[dict[str, object]] = {}

    @classmethod
    def _prepare_hamiltonian(cls, config: ModelConfig) -> dict[tuple[tuple[int, int], ...], complex]:
        def index(i: int, j: int, o: int) -> int:
            return (i + j * config.m) * 2 + o

        hamiltonian_dict: dict[tuple[tuple[int, int], ...], complex] = {}
        for i in range(config.m):
            for j in range(config.n):
                if i != 0:
                    hamiltonian_dict[(index(i, j, 0), 1), (index(i - 1, j, 0), 0)] = -config.t
                    hamiltonian_dict[(index(i - 1, j, 0), 1), (index(i, j, 0), 0)] = -config.t
                    hamiltonian_dict[(index(i, j, 1), 1), (index(i - 1, j, 1), 0)] = -config.t
                    hamiltonian_dict[(index(i - 1, j, 1), 1), (index(i, j, 1), 0)] = -config.t
                if j != 0:
                    hamiltonian_dict[(index(i, j, 0), 1), (index(i, j - 1, 0), 0)] = -config.t
                    hamiltonian_dict[(index(i, j - 1, 0), 1), (index(i, j, 0), 0)] = -config.t
                    hamiltonian_dict[(index(i, j, 1), 1), (index(i, j - 1, 1), 0)] = -config.t
                    hamiltonian_dict[(index(i, j - 1, 1), 1), (index(i, j, 1), 0)] = -config.t

                hamiltonian_dict[
                    (index(i, j, 0), 1),
                    (index(i, j, 0), 0),
                    (index(i, j, 1), 1),
                    (index(i, j, 1), 0),
                ] = config.u

                hamiltonian_dict[(index(i, j, 0), 1), (index(i, j, 0), 0)] = -config.mu
                hamiltonian_dict[(index(i, j, 1), 1), (index(i, j, 1), 0)] = -config.mu
        return hamiltonian_dict

    def __init__(self, config: ModelConfig) -> None:
        assert config.electron_number is not None
        self.m: int = config.m
        self.n: int = config.n
        self.electron_number: int = config.electron_number
        self.n_qubits: int = config.m * config.n * 2
        self.ref_energy: float = config.ref_energy
        logger.info(
            "Constructing Hubbard model: %dx%d, t=%.4f, U=%.4f, mu=%.4f, N=%d",
            config.m,
            config.n,
            config.t,
            config.u,
            config.mu,
            config.electron_number,
        )
        hamiltonian_dict = self._prepare_hamiltonian(config)
        self.hamiltonian = FermiHamiltonian(
            hamiltonian_dict,
            n_qubits=self.n_qubits,
            devices=config.devices,
        )

    def compute_diagonal_within_subspace(self, configs: jax.Array) -> jax.Array:
        return self.hamiltonian.compute_diagonal_within_subspace(configs)

    def apply_within_subspace(
        self,
        configs_i: jax.Array,
        psi_i: jax.Array,
        configs_j: jax.Array,
        *,
        direction: int = 0,
    ) -> jax.Array:
        return self.hamiltonian.apply_within_subspace(configs_i, psi_i, configs_j, direction=direction)

    def find_all_relative_configs(
        self,
        configs_i: jax.Array,
        psi_i: jax.Array,
        configs_exclude: jax.Array | None = None,
        *,
        hash_capacity: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        n_qubytes = (self.n_qubits + 7) // 8
        exclude = jnp.zeros((0, n_qubytes), dtype=jnp.uint8) if configs_exclude is None else configs_exclude
        return self.hamiltonian.find_all_relative_configs(configs_i, psi_i, exclude, hash_capacity=hash_capacity)

    def find_topk_relative_configs(
        self,
        configs_i: jax.Array,
        psi_i: jax.Array,
        count_selected: int,
        configs_exclude: jax.Array | None = None,
    ) -> jax.Array:
        n_qubytes = (self.n_qubits + 7) // 8
        exclude = jnp.zeros((0, n_qubytes), dtype=jnp.uint8) if configs_exclude is None else configs_exclude
        return self.hamiltonian.find_topk_relative_configs(configs_i, psi_i, count_selected, exclude)

    def show_config(self, config: jax.Array) -> str:
        bits = "".join(f"{int(byte):08b}"[::-1] for byte in config)
        rows = []
        for j in range(self.n):
            cells = []
            for i in range(self.m):
                start = (i + j * self.m) * 2
                cells.append(self._show_config_site(bits[start : start + 2]))
            rows.append("".join(cells))
        return "[" + ".".join(rows) + "]"

    @staticmethod
    def _show_config_site(occupation: str) -> str:
        match occupation:
            case "00":
                return " "
            case "10":
                return "↑"
            case "01":
                return "↓"
            case "11":
                return "↕"
            case _:
                raise ValueError(f"Invalid occupation string: {occupation}")


model_dict["hubbard"] = Model
