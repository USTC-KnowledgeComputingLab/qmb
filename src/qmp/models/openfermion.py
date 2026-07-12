"""OpenFermion MolecularData model."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import ClassVar

import jax
import jax.numpy as jnp
import openfermion

from qmp.hamiltonian.fermi_hamiltonian import FermiHamiltonian

from ._model import ModelProto, model_dict

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelConfig:
    """Configuration for the OpenFermion model."""

    model_path: str
    devices: list[str] = dataclasses.field(default_factory=lambda: ["localhost:cpu:0"])


class Model(ModelProto[ModelConfig]):
    """OpenFermion MolecularData model wrapping a FermiHamiltonian."""

    network_dict: ClassVar[dict[str, object]] = {}

    def __init__(self, config: ModelConfig) -> None:
        model_path = pathlib.Path(config.model_path)
        logger.info("Loading OpenFermion model from file: %s", model_path)
        molecule = openfermion.MolecularData(filename=str(model_path.resolve()))

        n_qubits = molecule.n_qubits
        fci_energy = molecule.fci_energy
        if n_qubits is None or fci_energy is None:
            raise ValueError(f"MolecularData at {model_path} is missing n_qubits or fci_energy.")

        self.n_qubits: int = int(n_qubits)
        self.n_electrons: int = int(molecule.n_electrons)
        self.n_spins: int = int(molecule.multiplicity) - 1
        self.ref_energy: float = float(fci_energy)
        logger.info(
            "Identified %d qubits, %d electrons, n_spins=%d, fci_energy=%.10f",
            self.n_qubits,
            self.n_electrons,
            self.n_spins,
            self.ref_energy,
        )

        fermion_operator = openfermion.transforms.get_fermion_operator(molecule.get_molecular_hamiltonian())
        hamiltonian_dict = {key: complex(value) for key, value in fermion_operator.terms.items()}
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
        cells = [self._show_config_site(bits[index : index + 2]) for index in range(0, self.n_qubits, 2)]
        return "[" + "".join(cells) + "]"

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


model_dict["openfermion"] = Model
