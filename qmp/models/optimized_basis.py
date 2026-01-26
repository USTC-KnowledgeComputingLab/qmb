"""
This file implements a model that loads the Hamiltonian from an optimized basis representation.
"""

import logging
import dataclasses
import pathlib
import torch
from ..hamiltonian import Hamiltonian
from ..utility.model_dict import model_dict, ModelProto, NetworkConfigProto, NetworkProto
from ..networks.mlp import WaveFunctionElectronUpDown as MlpWaveFunction
from ..networks.mlp import WaveFunctionElectron as MlpWaveFunctionElectron
from ..networks.transformers import WaveFunctionElectronUpDown as TransformersWaveFunction
from ..networks.transformers import WaveFunctionElectron as TransformersWaveFunctionElectron


@dataclasses.dataclass
class ModelConfig:
    """
    The configuration of the model.
    """

    # The path of the model file
    model_path: pathlib.Path
    # The reference energy
    ref_energy: float = 0.0


class Model(ModelProto[ModelConfig]):
    """
    This class handles the models loaded from optimized basis tensors.
    """

    network_dict: dict[str, type[NetworkConfigProto["Model"]]] = {}

    config_t = ModelConfig

    def __init__(self, args: ModelConfig) -> None:
        model_path = args.model_path
        logging.info("Loading Optimized Basis Hamiltonian from file: %s", model_path)

        # Load the data dictionary
        data = torch.load(model_path, map_location="cpu", weights_only=True)

        if not isinstance(data, dict) or "hamiltonian" not in data:
            raise ValueError(f"Invalid optimized basis file format at {model_path}")

        self.hamiltonian = Hamiltonian(data["hamiltonian"], kind="fermi")
        self.n_qubits = data["n_qubits"]
        self.n_electrons = data["n_electrons"]
        self.n_spins = data["n_spins"]
        self.ref_energy = args.ref_energy or data.get("ref_energy", 0.0)

        # Store the unitary transformation matrix for future use
        self.orbit_unitary: torch.Tensor | None = data.get("orbit_unitary")

        logging.info("Hamiltonian loaded successfully")
        logging.info(
            "System info: %d qubits, %d electrons, %d spin, ref_energy: %.10f",
            self.n_qubits,
            self.n_electrons,
            self.n_spins,
            self.ref_energy,
        )
        if self.orbit_unitary is not None:
            logging.info("Orbit unitary matrix loaded (shape: %s)", self.orbit_unitary.shape)

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

    def list_relative(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        configs_exclude: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hamiltonian.list_relative(configs_i, psi_i, configs_exclude)

    def diagonal_term(self, configs: torch.Tensor) -> torch.Tensor:
        return self.hamiltonian.diagonal_term(configs)

    def show_config(self, config: torch.Tensor) -> str:
        string = "".join(f"{i:08b}"[::-1] for i in config.cpu().numpy())
        return (
            "["
            + "".join(self._show_config_site(string[index : index + 2]) for index in range(0, self.n_qubits, 2))
            + "]"
        )

    def _show_config_site(self, string: str) -> str:
        match string:
            case "00":
                return " "
            case "10":
                return "↑"
            case "01":
                return "↓"
            case "11":
                return "↕"
            case _:
                raise ValueError(f"Invalid string: {string}")


model_dict["optimized_basis"] = Model


@dataclasses.dataclass
class MlpConfig:
    """
    The configuration of the MLP network.
    """

    # The hidden widths of the network
    hidden: tuple[int, ...] = (512,)

    def create(self, model: Model) -> NetworkProto:
        """
        Create a MLP network for the model.
        """
        logging.info("Hidden layer widths: %a", self.hidden)

        network = MlpWaveFunction(
            double_sites=model.n_qubits,
            physical_dim=2,
            is_complex=True,
            spin_up=(model.n_electrons + model.n_spins) // 2,
            spin_down=(model.n_electrons - model.n_spins) // 2,
            hidden_size=self.hidden,
            ordering=+1,
        )

        return network


Model.network_dict["mlp/u1u1"] = MlpConfig
Model.network_dict["mlp"] = MlpConfig


@dataclasses.dataclass
class TransformersConfig:
    """
    The configuration of the transformers network.
    """

    # Embedding dimension
    embedding_dim: int = 512
    # Heads number
    heads_num: int = 8
    # Feedforward dimension
    feed_forward_dim: int = 2048
    # Shared expert number
    shared_expert_num: int = 1
    # Routed expert number
    routed_expert_num: int = 0
    # Selected expert number
    selected_expert_num: int = 0
    # Network depth
    depth: int = 6

    def create(self, model: Model) -> NetworkProto:
        """
        Create a transformers network for the model.
        """
        logging.info(
            "Transformers network configuration: "
            "embedding dimension: %d, "
            "number of heads: %d, "
            "feed-forward dimension: %d, "
            "shared expert number: %d, "
            "routed expert number: %d, "
            "selected expert number: %d, "
            "depth: %d",
            self.embedding_dim,
            self.heads_num,
            self.feed_forward_dim,
            self.shared_expert_num,
            self.routed_expert_num,
            self.selected_expert_num,
            self.depth,
        )

        network = TransformersWaveFunction(
            double_sites=model.n_qubits,
            physical_dim=2,
            is_complex=True,
            spin_up=(model.n_electrons + model.n_spins) // 2,
            spin_down=(model.n_electrons - model.n_spins) // 2,
            embedding_dim=self.embedding_dim,
            heads_num=self.heads_num,
            feed_forward_dim=self.feed_forward_dim,
            shared_num=self.shared_expert_num,
            routed_num=self.routed_expert_num,
            selected_num=self.selected_expert_num,
            depth=self.depth,
            ordering=+1,
        )

        return network


Model.network_dict["transformers/u1u1"] = TransformersConfig
Model.network_dict["transformers"] = TransformersConfig


@dataclasses.dataclass
class MlpElectronConfig:
    """
    The configuration of the MLP network with total electron number conservation.
    """

    # The hidden widths of the network
    hidden: tuple[int, ...] = (512,)

    def create(self, model: Model) -> NetworkProto:
        """
        Create a MLP network for the model.
        """
        logging.info("Hidden layer widths: %a", self.hidden)

        network = MlpWaveFunctionElectron(
            sites=model.n_qubits,
            physical_dim=2,
            is_complex=True,
            electrons=model.n_electrons,
            hidden_size=self.hidden,
            ordering=+1,
        )

        return network


Model.network_dict["mlp/u1"] = MlpElectronConfig


@dataclasses.dataclass
class TransformersElectronConfig:
    """
    The configuration of the transformers network with total electron number conservation.
    """

    # Embedding dimension
    embedding_dim: int = 512
    # Heads number
    heads_num: int = 8
    # Feedforward dimension
    feed_forward_dim: int = 2048
    # Shared expert number
    shared_expert_num: int = 1
    # Routed expert number
    routed_expert_num: int = 0
    # Selected expert number
    selected_expert_num: int = 0
    # Network depth
    depth: int = 6

    def create(self, model: Model) -> NetworkProto:
        """
        Create a transformers network for the model.
        """
        logging.info(
            "Transformers network configuration: "
            "embedding dimension: %d, "
            "number of heads: %d, "
            "feed-forward dimension: %d, "
            "shared expert number: %d, "
            "routed expert number: %d, "
            "selected expert number: %d, "
            "depth: %d",
            self.embedding_dim,
            self.heads_num,
            self.feed_forward_dim,
            self.shared_expert_num,
            self.routed_expert_num,
            self.selected_expert_num,
            self.depth,
        )

        network = TransformersWaveFunctionElectron(
            sites=model.n_qubits,
            physical_dim=2,
            is_complex=True,
            electrons=model.n_electrons,
            embedding_dim=self.embedding_dim,
            heads_num=self.heads_num,
            feed_forward_dim=self.feed_forward_dim,
            shared_num=self.shared_expert_num,
            routed_num=self.routed_expert_num,
            selected_num=self.selected_expert_num,
            depth=self.depth,
            ordering=+1,
        )

        return network


Model.network_dict["transformers/u1"] = TransformersElectronConfig
