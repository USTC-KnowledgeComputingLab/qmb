"""
This file offers an interface for defining Ising-like models on a two-dimensional lattice.
"""

import logging
import dataclasses
import collections
import torch
from ..networks.mlp import WaveFunctionNormal as MlpWaveFunction
from ..networks.mlp import WaveFunctionElectron as MlpWaveFunctionElectron
from ..networks.transformers import WaveFunctionNormal as TransformersWaveFunction
from ..networks.transformers import WaveFunctionElectron as TransformersWaveFunctionElectron
from ..hamiltonian import Hamiltonian
from ..utility.model_dict import model_dict, ModelProto, NetworkProto, NetworkConfigProto


@dataclasses.dataclass
class ModelConfig:
    """
    The configuration for the Ising-like model.
    """

    # The width of the ising lattice
    m: int
    # The height of the ising lattice
    n: int

    # The coefficient of X
    x: float = 0
    # The coefficient of Y
    y: float = 0
    # The coefficient of Z
    z: float = 0
    # The coefficient of XX for horizontal bond
    xh: float = 0
    # The coefficient of YY for horizontal bond
    yh: float = 0
    # The coefficient of ZZ for horizontal bond
    zh: float = 0
    # The coefficient of XX for vertical bond
    xv: float = 0
    # The coefficient of YY for vertical bond
    yv: float = 0
    # The coefficient of ZZ for vertical bond
    zv: float = 0
    # The coefficient of XX for diagonal bond
    xd: float = 0
    # The coefficient of YY for diagonal bond
    yd: float = 0
    # The coefficient of ZZ for diagonal bond
    zd: float = 0
    # The coefficient of XX for antidiagonal bond
    xa: float = 0
    # The coefficient of YY for antidiagonal bond
    ya: float = 0
    # The coefficient of ZZ for antidiagonal bond
    za: float = 0

    # The ref energy of the model
    ref_energy: float = 0


class Model(ModelProto[ModelConfig]):
    """
    This class handles the Ising-like model.
    """

    network_dict: dict[str, type[NetworkConfigProto["Model"]]] = {}

    config_t = ModelConfig

    @classmethod
    def _prepare_hamiltonian(cls, args: ModelConfig) -> dict[tuple[tuple[int, int], ...], complex]:
        def _index(i: int, j: int) -> int:
            return i + j * args.m

        def _x(i: int, j: int) -> tuple[tuple[tuple[tuple[int, int], ...], complex], ...]:
            return (
                (((_index(i, j), 1),), +1),
                (((_index(i, j), 0),), +1),
            )

        def _y(i: int, j: int) -> tuple[tuple[tuple[tuple[int, int], ...], complex], ...]:
            return (
                (((_index(i, j), 1),), -1j),
                (((_index(i, j), 0),), +1j),
            )

        def _z(i: int, j: int) -> tuple[tuple[tuple[tuple[int, int], ...], complex], ...]:
            return (
                (((_index(i, j), 1), (_index(i, j), 0)), +1),
                (((_index(i, j), 0), (_index(i, j), 1)), -1),
            )

        hamiltonian: dict[tuple[tuple[int, int], ...], complex] = collections.defaultdict(lambda: 0)
        # Express spin pauli matrix in hard core boson language.
        for i in range(args.m):
            for j in range(args.n):
                k: tuple[tuple[int, int], ...]
                k1: tuple[tuple[int, int], ...]
                k2: tuple[tuple[int, int], ...]
                v: complex
                v1: complex
                v2: complex
                if True:
                    if args.x != 0:
                        for k, v in _x(i, j):
                            hamiltonian[k] += v * args.x
                    if args.y != 0:
                        for k, v in _y(i, j):
                            hamiltonian[k] += v * args.y
                    if args.z != 0:
                        for k, v in _z(i, j):
                            hamiltonian[k] += v * args.z
                if i != 0:
                    if args.xh != 0:
                        for k1, v1 in _x(i, j):
                            for k2, v2 in _x(i - 1, j):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.xh
                    if args.yh != 0:
                        for k1, v1 in _y(i, j):
                            for k2, v2 in _y(i - 1, j):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.yh
                    if args.zh != 0:
                        for k1, v1 in _z(i, j):
                            for k2, v2 in _z(i - 1, j):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.zh
                if j != 0:
                    if args.xv != 0:
                        for k1, v1 in _x(i, j):
                            for k2, v2 in _x(i, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.xv
                    if args.yv != 0:
                        for k1, v1 in _y(i, j):
                            for k2, v2 in _y(i, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.yv
                    if args.zv != 0:
                        for k1, v1 in _z(i, j):
                            for k2, v2 in _z(i, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.zv
                if i != 0 and j != 0:
                    if args.xd != 0:
                        for k1, v1 in _x(i, j):
                            for k2, v2 in _x(i - 1, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.xd
                    if args.yd != 0:
                        for k1, v1 in _y(i, j):
                            for k2, v2 in _y(i - 1, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.yd
                    if args.zd != 0:
                        for k1, v1 in _z(i, j):
                            for k2, v2 in _z(i - 1, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.zd
                if i != 0 and j != args.n - 1:
                    if args.xa != 0:
                        for k1, v1 in _x(i, j):
                            for k2, v2 in _x(i - 1, j + 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.xa
                    if args.ya != 0:
                        for k1, v1 in _y(i, j):
                            for k2, v2 in _y(i - 1, j + 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.ya
                    if args.za != 0:
                        for k1, v1 in _z(i, j):
                            for k2, v2 in _z(i - 1, j + 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.za
        return hamiltonian

    def __init__(self, args: ModelConfig) -> None:
        self.m: int = args.m
        self.n: int = args.n
        logging.info(
            "Constructing Ising model: width = %d, height = %d, ref_energy = %.4f", self.m, self.n, args.ref_energy
        )

        logging.info("Initializing Ising Hamiltonian for the lattice")
        hamiltonian_dict: dict[tuple[tuple[int, int], ...], complex] = self._prepare_hamiltonian(args)
        logging.info("Hamiltonian dictionary initialized successfully.")

        self.ref_energy: float = args.ref_energy

        logging.info("Converting the Hamiltonian to internal Hamiltonian representation")
        self.hamiltonian: Hamiltonian = Hamiltonian(hamiltonian_dict, kind="bose2")
        logging.info("Internal Hamiltonian representation successfully created.")

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
            + ".".join(
                "".join("↑" if string[i + j * self.m] == "0" else "↓" for i in range(self.m)) for j in range(self.n)
            )
            + "]"
        )


model_dict["ising"] = Model


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
            sites=model.m * model.n,
            physical_dim=2,
            is_complex=True,
            hidden_size=self.hidden,
            ordering=+1,
        )

        return network


Model.network_dict["mlp/0"] = MlpConfig
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
            sites=model.m * model.n,
            physical_dim=2,
            is_complex=True,
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


Model.network_dict["transformers/0"] = TransformersConfig
Model.network_dict["transformers"] = TransformersConfig


@dataclasses.dataclass
class MlpElectronConfig:
    """
    The configuration of the MLP network with electron number conservation.
    """

    # The hidden widths of the network
    hidden: tuple[int, ...] = (512,)

    def create(self, model: Model) -> NetworkProto:
        """
        Create a MLP network for the model.
        """
        logging.info("Hidden layer widths: %a", self.hidden)

        electrons = (model.m * model.n) // 2

        network = MlpWaveFunctionElectron(
            sites=model.m * model.n,
            physical_dim=2,
            is_complex=True,
            electrons=electrons,
            hidden_size=self.hidden,
            ordering=+1,
        )

        return network


Model.network_dict["mlp/u1"] = MlpElectronConfig


@dataclasses.dataclass
class TransformersElectronConfig:
    """
    The configuration of the transformers network with electron number conservation.
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

        electrons = (model.m * model.n) // 2

        network = TransformersWaveFunctionElectron(
            sites=model.m * model.n,
            physical_dim=2,
            is_complex=True,
            electrons=electrons,
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
