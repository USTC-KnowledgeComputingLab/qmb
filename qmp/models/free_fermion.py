"""
This file offers an interface for defining free fermion models on a two-dimensional lattice.
"""

import logging
import dataclasses
import torch
from ..hamiltonian import Hamiltonian
from ..utility.model_dict import model_dict, ModelProto, NetworkConfigProto


@dataclasses.dataclass
class ModelConfig:
    """
    The configuration for the free fermion model.
    """

    # The width of the free fermion lattice
    m: int
    # The height of the free fermion lattice
    n: int

    # The electron number
    electron_number: int

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0:
            raise ValueError("The dimensions of the free fermion model must be positive integers.")

        if self.electron_number < 0 or self.electron_number > self.m * self.n:
            raise ValueError(
                f"The electron number {self.electron_number} is out of bounds for a {self.m}x{self.n} lattice. Each site can host up to one electron."
            )


class Model(ModelProto[ModelConfig]):
    """
    This class handles the free fermion model.
    """

    network_dict: dict[str, type[NetworkConfigProto["Model"]]] = {}

    config_t = ModelConfig

    @classmethod
    def _prepare_hamiltonian(cls, args: ModelConfig) -> dict[tuple[tuple[int, int], ...], complex]:
        def _index(i: int, j: int) -> int:
            return i + j * args.m

        hamiltonian_dict: dict[tuple[tuple[int, int], ...], complex] = {}
        for i in range(args.m):
            for j in range(args.n):
                # Nearest neighbor hopping
                if i != 0:
                    hamiltonian_dict[(_index(i, j), 1), (_index(i - 1, j), 0)] = 1
                    hamiltonian_dict[(_index(i - 1, j), 1), (_index(i, j), 0)] = 1
                if j != 0:
                    hamiltonian_dict[(_index(i, j), 1), (_index(i, j - 1), 0)] = 1
                    hamiltonian_dict[(_index(i, j - 1), 1), (_index(i, j), 0)] = 1

        return hamiltonian_dict

    def _calculate_ref_energy(self, hamiltonian_dict: dict[tuple[tuple[int, int], ...], complex]) -> float:
        sites = self.m * self.n
        hamiltonian = torch.zeros((sites, sites), dtype=torch.complex128)
        for ((site_1, _), (site_2, _)), value in hamiltonian_dict.items():
            hamiltonian[site_1, site_2] = torch.tensor(value, dtype=torch.complex128)
        return torch.linalg.eigh(hamiltonian).eigenvalues[: self.electron_number].sum().item()  # pylint: disable=not-callable

    def __init__(
        self,
        args: ModelConfig,
    ):
        assert args.electron_number is not None
        self.m: int = args.m
        self.n: int = args.n
        self.electron_number: int = args.electron_number
        logging.info("Constructing a free fermion model with dimensions: width = %d, height = %d", self.m, self.n)
        logging.info("The parameters of the model are: N = %d", args.electron_number)

        logging.info("Initializing Hamiltonian for the lattice")
        hamiltonian_dict: dict[tuple[tuple[int, int], ...], complex] = self._prepare_hamiltonian(args)
        logging.info("Hamiltonian initialization complete")

        self.ref_energy: float = self._calculate_ref_energy(hamiltonian_dict)
        logging.info("The ref energy is %.10f", self.ref_energy)

        logging.info("Converting the Hamiltonian to internal Hamiltonian representation")
        self.hamiltonian: Hamiltonian = Hamiltonian(hamiltonian_dict, kind="fermi")
        logging.info("Internal Hamiltonian representation for model has been successfully created")

    def apply_within(self, configs_i: torch.Tensor, psi_i: torch.Tensor, configs_j: torch.Tensor) -> torch.Tensor:
        return self.hamiltonian.apply_within(configs_i, psi_i, configs_j)

    def find_relative(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        count_selected: int,
        configs_exclude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.hamiltonian.find_relative(configs_i, psi_i, count_selected, configs_exclude)

    def diagonal_term(self, configs: torch.Tensor) -> torch.Tensor:
        return self.hamiltonian.diagonal_term(configs)

    def show_config(self, config: torch.Tensor) -> str:
        string = "".join(f"{i:08b}"[::-1] for i in config.cpu().numpy())
        return (
            "["
            + ".".join(
                "".join("-" if string[i + j * self.m] == "0" else "x" for i in range(self.m)) for j in range(self.n)
            )
            + "]"
        )


model_dict["free_fermion"] = Model
