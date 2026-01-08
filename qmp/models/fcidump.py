"""
This file provides an interface to work with FCIDUMP files.
"""

import typing
import logging
import dataclasses
import re
import gzip
import pathlib
import hashlib
import torch
import yaml
import openfermion
import platformdirs
from ..networks.mlp import WaveFunctionElectronUpDown as MlpWaveFunction
from ..networks.attention import WaveFunctionElectronUpDown as AttentionWaveFunction
from ..networks.crossmlp import WaveFunction as CrossMlpWaveFunction
from ..hamiltonian import Hamiltonian
from ..utility.model_dict import model_dict, ModelProto, NetworkProto, NetworkConfigProto


@dataclasses.dataclass
class ModelConfig:
    """
    The configuration of the model.
    """

    # The openfermion model name
    model_name: str
    # The path of models folder
    model_path: pathlib.Path
    # The ref energy of the model, leave empty to read from FCIDUMP.yaml
    ref_energy: float | None = None

    def __post_init__(self) -> None:
        self.model_path = pathlib.Path(self.model_path)


def _read_fcidump(
    file_name: pathlib.Path, *, cached: bool = False
) -> tuple[tuple[int, int, int], dict[tuple[tuple[int, int], ...], complex]]:
    # pylint: disable=too-many-locals
    with (
        gzip.open(file_name, "rt", encoding="utf-8")
        if file_name.name.endswith(".gz")
        else open(file_name, "rt", encoding="utf-8") as file
    ):
        n_orbit: int | None = None
        n_electron: int | None = None
        n_spin: int | None = None
        for line in file:
            data: str = line.lower()
            if (match := re.search(r"norb\s*=\s*(\d+)", data)) is not None:
                n_orbit = int(match.group(1))
            if (match := re.search(r"nelec\s*=\s*(\d+)", data)) is not None:
                n_electron = int(match.group(1))
            if (match := re.search(r"ms2\s*=\s*(\d+)", data)) is not None:
                n_spin = int(match.group(1))
            if "&end" in data:
                break
        assert n_orbit is not None
        assert n_electron is not None
        assert n_spin is not None
        if cached:
            return (n_orbit, n_electron, n_spin), {}
        energy_0: float = 0.0
        energy_1: torch.Tensor = torch.zeros([n_orbit, n_orbit], dtype=torch.float64)
        energy_2: torch.Tensor = torch.zeros([n_orbit, n_orbit, n_orbit, n_orbit], dtype=torch.float64)
        for line in file:
            pieces: list[str] = line.split()
            coefficient: float = float(pieces[0])
            sites: tuple[int, ...] = tuple(int(i) - 1 for i in pieces[1:])
            match sites:
                case (-1, -1, -1, -1):
                    energy_0 = coefficient
                case (_, -1, -1, -1):
                    # Psi4 writes additional non-standard one-electron integrals in this format, which we omit.
                    pass
                case (i, j, -1, -1):
                    energy_1[i, j] = coefficient
                    energy_1[j, i] = coefficient
                case (_, _, _, -1):
                    # In the standard FCIDUMP format, there is no such term.
                    raise ValueError(f"Invalid FCIDUMP format: {sites}")
                case (i, j, k, l):
                    energy_2[i, j, k, l] = coefficient
                    energy_2[i, j, l, k] = coefficient
                    energy_2[j, i, k, l] = coefficient
                    energy_2[j, i, l, k] = coefficient
                    energy_2[l, k, j, i] = coefficient
                    energy_2[k, l, j, i] = coefficient
                    energy_2[l, k, i, j] = coefficient
                    energy_2[k, l, i, j] = coefficient
                case _:
                    raise ValueError(f"Invalid FCIDUMP format: {sites}")

    energy_2 = energy_2.permute(0, 2, 3, 1).contiguous() / 2
    energy_1_b: torch.Tensor = torch.zeros([n_orbit * 2, n_orbit * 2], dtype=torch.float64)
    energy_2_b: torch.Tensor = torch.zeros([n_orbit * 2, n_orbit * 2, n_orbit * 2, n_orbit * 2], dtype=torch.float64)
    energy_1_b[0::2, 0::2] = energy_1
    energy_1_b[1::2, 1::2] = energy_1
    energy_2_b[0::2, 0::2, 0::2, 0::2] = energy_2
    energy_2_b[0::2, 1::2, 1::2, 0::2] = energy_2
    energy_2_b[1::2, 0::2, 0::2, 1::2] = energy_2
    energy_2_b[1::2, 1::2, 1::2, 1::2] = energy_2

    interaction_operator: openfermion.InteractionOperator = openfermion.InteractionOperator(
        energy_0, energy_1_b.numpy(), energy_2_b.numpy()
    )  # type: ignore[no-untyped-call]
    fermion_operator: openfermion.FermionOperator = openfermion.get_fermion_operator(interaction_operator)  # type: ignore[no-untyped-call]
    return (n_orbit, n_electron, n_spin), {
        k: complex(v) for k, v in openfermion.normal_ordered(fermion_operator).terms.items()
    }  # type: ignore[no-untyped-call]


class Model(ModelProto[ModelConfig]):
    """
    This class handles the models from FCIDUMP files.
    """

    network_dict: dict[str, type[NetworkConfigProto["Model"]]] = {}

    config_t = ModelConfig

    def __init__(self, args: ModelConfig) -> None:
        # pylint: disable=too-many-locals
        logging.info("Input arguments successfully parsed")
        logging.info("Model name: %s, Model path: %s", args.model_name, args.model_path)

        model_name = args.model_name
        model_path = args.model_path
        ref_energy = args.ref_energy
        assert model_path is not None

        model_file_name = model_path / f"{model_name}.FCIDUMP.gz"
        model_file_name = model_file_name if model_file_name.exists() else model_path / model_name

        checksum = hashlib.sha256(model_file_name.read_bytes()).hexdigest() + "v5"
        cache_file = platformdirs.user_cache_path("qmp", "kclab") / checksum
        if cache_file.exists():
            logging.info("Loading FCIDUMP metadata '%s' from file: %s", model_name, model_file_name)
            (n_orbit, n_electron, n_spin), _ = _read_fcidump(model_file_name, cached=True)
            logging.info("FCIDUMP metadata '%s' successfully loaded", model_name)

            logging.info("Loading FCIDUMP Hamiltonian '%s' from cache", model_name)
            openfermion_hamiltonian_data = torch.load(cache_file, map_location="cpu", weights_only=True)
            logging.info("FCIDUMP Hamiltonian '%s' successfully loaded", model_name)

            logging.info("Recovering internal Hamiltonian representation for model '%s'", model_name)
            self.hamiltonian = Hamiltonian(openfermion_hamiltonian_data, kind="fermi")
            logging.info("Internal Hamiltonian representation for model '%s' successfully recovered", model_name)
        else:
            logging.info("Loading FCIDUMP Hamiltonian '%s' from file: %s", model_name, model_file_name)
            (n_orbit, n_electron, n_spin), openfermion_hamiltonian_dict = _read_fcidump(model_file_name)
            logging.info("FCIDUMP Hamiltonian '%s' successfully loaded", model_name)

            logging.info("Converting OpenFermion Hamiltonian to internal Hamiltonian representation")
            self.hamiltonian = Hamiltonian(openfermion_hamiltonian_dict, kind="fermi")
            logging.info("Internal Hamiltonian representation for model '%s' has been successfully created", model_name)

            logging.info("Caching OpenFermion Hamiltonian for model '%s'", model_name)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save((self.hamiltonian.site, self.hamiltonian.kind, self.hamiltonian.coef), cache_file)
            logging.info("OpenFermion Hamiltonian for model '%s' successfully cached", model_name)

        self.n_qubit: int = n_orbit * 2
        self.n_electron: int = n_electron
        self.n_spin: int = n_spin
        logging.info(
            "Identified %d qubits, %d electrons and %d spin for model '%s'",
            self.n_qubit,
            self.n_electron,
            self.n_spin,
            model_name,
        )

        self.ref_energy: float
        if ref_energy is not None:
            self.ref_energy = ref_energy
        else:
            fcidump_ref_energy_file = model_file_name.parent / "FCIDUMP.yaml"
            if fcidump_ref_energy_file.exists():
                with open(fcidump_ref_energy_file, "rt", encoding="utf-8") as file:
                    fcidump_ref_energy_data = yaml.safe_load(file)
                self.ref_energy = fcidump_ref_energy_data.get(pathlib.Path(model_name).name, 0)
            else:
                self.ref_energy = 0
        logging.info("Reference energy for model '%s' is %.10f", model_name, self.ref_energy)

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
            + "".join(self._show_config_site(string[index : index + 2]) for index in range(0, self.n_qubit, 2))
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


model_dict["fcidump"] = Model


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
            double_sites=model.n_qubit,
            physical_dim=2,
            is_complex=True,
            spin_up=(model.n_electron + model.n_spin) // 2,
            spin_down=(model.n_electron - model.n_spin) // 2,
            hidden_size=self.hidden,
            ordering=+1,
        )

        return network


Model.network_dict["mlp"] = MlpConfig


@dataclasses.dataclass
class AttentionConfig:
    """
    The configuration of the attention network.
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
        Create an attention network for the model.
        """
        logging.info(
            "Attention network configuration: "
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

        network = AttentionWaveFunction(
            double_sites=model.n_qubit,
            physical_dim=2,
            is_complex=True,
            spin_up=(model.n_electron + model.n_spin) // 2,
            spin_down=(model.n_electron - model.n_spin) // 2,
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


Model.network_dict["attention"] = AttentionConfig


@dataclasses.dataclass
class CrossMlpConfig:
    """
    The configuration of the cross MLP network.
    """

    # The hidden widths of the embedding subnetwork
    embedding_hidden: tuple[int, ...] = (64,)
    # The dimension of the embedding
    embedding_size: int = 16
    # The hidden widths of the momentum subnetwork
    momentum_hidden: tuple[int, ...] = (64,)
    # The number of max momentum order
    momentum_count: int = 1
    # The hidden widths of the tail part
    tail_hidden: tuple[int, ...] = (64,)
    # The kind of the crossmlp forward function
    kind: typing.Literal[0, 1, 2] = 0
    # The ordering of the sites
    ordering: int | list[int] = +1

    def create(self, model: Model) -> NetworkProto:
        """
        Create a cross MLP network for the model.
        """
        logging.info(
            "Cross MLP network configuration: "
            "embedding hidden widths: %a, "
            "embedding size: %d, "
            "momentum hidden widths: %a, "
            "momentum count: %d, "
            "tail hidden widths: %a, "
            "kind: %d, "
            "ordering: %s",
            self.embedding_hidden,
            self.embedding_size,
            self.momentum_hidden,
            self.momentum_count,
            self.tail_hidden,
            self.kind,
            self.ordering,
        )

        network = CrossMlpWaveFunction(
            sites=model.n_qubit,
            physical_dim=2,
            is_complex=False,
            embedding_hidden_size=self.embedding_hidden,
            embedding_size=self.embedding_size,
            momentum_hidden_size=self.momentum_hidden,
            momentum_count=self.momentum_count,
            tail_hidden_size=self.tail_hidden,
            kind=self.kind,
            ordering=self.ordering,
        )

        return network


Model.network_dict["crossmlp"] = CrossMlpConfig
