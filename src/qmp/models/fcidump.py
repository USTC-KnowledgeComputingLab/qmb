"""FCIDUMP file model."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import logging
import pathlib
import pickle
import re
from typing import TYPE_CHECKING, ClassVar

import dacite
import jax
import jax.numpy as jnp
import numpy as np
import openfermion
import platformdirs

from qmp.hamiltonian.fermi_hamiltonian import FermiHamiltonian
from qmp.networks.mlp import WaveFunctionElectron as MlpWaveFunctionElectron
from qmp.networks.mlp import WaveFunctionElectronUpDown as MlpWaveFunctionUpDown
from qmp.networks.transformers import WaveFunctionElectron as TransformersWaveFunctionElectron
from qmp.networks.transformers import WaveFunctionElectronUpDown as TransformersWaveFunctionUpDown

from ._model import ModelProto, model_config_dict, model_dict

if TYPE_CHECKING:
    from flax import nnx

    from qmp.networks._protocol import NetworkProto

logger = logging.getLogger(__name__)

_CACHE_VERSION = "v1"


def _cache_dir() -> pathlib.Path:
    """Return the fcidump cache directory (redirectable in tests via monkeypatch).

    Uses the same ("qmp", "kclab") appname/appauthor as the CUDA loader for a
    single shared cache base. On Linux appauthor "kclab" is ignored (legacy);
    the models/fcidump subdirectory keeps this cache separate from the CUDA
    hamiltonian/fermi/ subtree.
    """
    return platformdirs.user_cache_path("qmp", "kclab") / "models" / "fcidump"


def read_fcidump(
    file_name: pathlib.Path,
    *,
    headonly: bool = False,
) -> tuple[tuple[int, int, int], dict[tuple[tuple[int, int], ...], complex]]:
    """Parse an FCIDUMP file into (n_orbit, n_electron, n_spin) and a Hamiltonian dict."""
    opener = gzip.open if file_name.name.endswith(".gz") else open
    with opener(file_name, "rt", encoding="utf-8") as file:
        n_orbit: int | None = None
        n_electron: int | None = None
        n_spin: int | None = None
        for line in file:
            data = line.lower()
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
        if headonly:
            return (n_orbit, n_electron, n_spin), {}

        energy_0 = 0.0
        energy_1 = np.zeros((n_orbit, n_orbit), dtype=np.float64)
        energy_2 = np.zeros((n_orbit, n_orbit, n_orbit, n_orbit), dtype=np.float64)
        for line in file:
            pieces = line.split()
            coefficient = float(pieces[0])
            sites = tuple(int(value) - 1 for value in pieces[1:])
            match sites:
                case (-1, -1, -1, -1):
                    energy_0 = coefficient
                case (_, -1, -1, -1):
                    pass
                case (i, j, -1, -1):
                    energy_1[i, j] = coefficient
                    energy_1[j, i] = coefficient
                case (_, _, _, -1):
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

    energy_2 = np.ascontiguousarray(energy_2.transpose(0, 2, 3, 1)) / 2
    energy_1_b = np.zeros((n_orbit * 2, n_orbit * 2), dtype=np.float64)
    energy_2_b = np.zeros((n_orbit * 2,) * 4, dtype=np.float64)
    energy_1_b[0::2, 0::2] = energy_1
    energy_1_b[1::2, 1::2] = energy_1
    energy_2_b[0::2, 0::2, 0::2, 0::2] = energy_2
    energy_2_b[0::2, 1::2, 1::2, 0::2] = energy_2
    energy_2_b[1::2, 0::2, 0::2, 1::2] = energy_2
    energy_2_b[1::2, 1::2, 1::2, 1::2] = energy_2

    interaction_operator = openfermion.InteractionOperator(energy_0, energy_1_b, energy_2_b)
    fermion_operator = openfermion.get_fermion_operator(interaction_operator)
    hamiltonian_dict = {
        key: complex(value) for key, value in openfermion.normal_ordered(fermion_operator).terms.items()
    }
    return (n_orbit, n_electron, n_spin), hamiltonian_dict


@dataclasses.dataclass
class ModelConfig:
    """Configuration for the FCIDUMP model."""

    model_path: str
    ref_energy: float = 0.0
    devices: list[str] = dataclasses.field(default_factory=lambda: ["localhost:cpu:0"])


class Model(ModelProto[ModelConfig]):
    """FCIDUMP file model wrapping a FermiHamiltonian."""

    network_dict: ClassVar[dict[str, object]] = {}

    def __init__(self, config: ModelConfig) -> None:
        model_path = pathlib.Path(config.model_path)
        checksum = hashlib.sha256(model_path.read_bytes()).hexdigest() + "-" + _CACHE_VERSION + ".pkl"
        cache_file = _cache_dir() / checksum

        if cache_file.exists():
            logger.info("Loading FCIDUMP metadata from file: %s", model_path)
            (n_orbit, n_electron, n_spin), _ = read_fcidump(model_path, headonly=True)
            logger.info("Loading FCIDUMP Hamiltonian from cache: %s", cache_file)
            with open(cache_file, "rb") as file:
                hamiltonian_dict = pickle.load(file)
        else:
            logger.info("Loading FCIDUMP Hamiltonian from file: %s", model_path)
            (n_orbit, n_electron, n_spin), hamiltonian_dict = read_fcidump(model_path)
            logger.info("Caching FCIDUMP Hamiltonian to: %s", cache_file)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as file:
                pickle.dump(hamiltonian_dict, file)

        self.n_qubits: int = n_orbit * 2
        self.n_electrons: int = n_electron
        self.n_spins: int = n_spin
        self.ref_energy: float = config.ref_energy
        logger.info(
            "Identified %d qubits, %d electrons, n_spins=%d, ref_energy=%.10f",
            self.n_qubits,
            self.n_electrons,
            self.n_spins,
            self.ref_energy,
        )
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

    def create_network(self, name: str, params: dict, *, rngs: nnx.Rngs) -> NetworkProto:
        cfg_cls = self.network_dict[name]
        cfg = dacite.from_dict(cfg_cls, params)  # ty: ignore — dacite from_dict stub is incorrect
        return cfg.create(self, rngs=rngs)

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


model_dict["fcidump"] = Model
model_config_dict["fcidump"] = ModelConfig


@dataclasses.dataclass
class MlpUpDownConfig:
    """MLP network with spin-up/spin-down electron-number conservation."""

    hidden_size: list[int] = dataclasses.field(default_factory=lambda: [512])
    ordering: int = 1

    def create(self, model: Model, *, rngs: nnx.Rngs) -> NetworkProto:
        """Build a spin-resolved MLP wave function for the model."""
        logger.info("Creating MLP (u1u1) network: hidden_size=%a", self.hidden_size)
        return MlpWaveFunctionUpDown(
            double_sites=model.n_qubits,
            spin_up=(model.n_electrons + model.n_spins) // 2,
            spin_down=(model.n_electrons - model.n_spins) // 2,
            hidden_size=self.hidden_size if isinstance(self.hidden_size, tuple) else tuple(self.hidden_size),
            ordering=self.ordering,
            rngs=rngs,
        )


Model.network_dict["mlp/u1u1"] = MlpUpDownConfig


@dataclasses.dataclass
class MlpElectronConfig:
    """MLP network with total electron-number conservation."""

    hidden_size: list[int] = dataclasses.field(default_factory=lambda: [512])
    ordering: int = 1

    def create(self, model: Model, *, rngs: nnx.Rngs) -> NetworkProto:
        """Build a total-electron MLP wave function for the model."""
        logger.info("Creating MLP (u1) network: hidden_size=%a", self.hidden_size)
        return MlpWaveFunctionElectron(
            sites=model.n_qubits,
            electrons=model.n_electrons,
            hidden_size=self.hidden_size if isinstance(self.hidden_size, tuple) else tuple(self.hidden_size),
            ordering=self.ordering,
            rngs=rngs,
        )


Model.network_dict["mlp/u1"] = MlpElectronConfig


@dataclasses.dataclass
class TransformersUpDownConfig:
    """Transformer network with spin-up/spin-down electron-number conservation."""

    embedding_dim: int = 512
    heads_num: int = 8
    feed_forward_dim: int = 2048
    depth: int = 6
    tail_hidden_dim: int = 512
    ordering: int = 1

    def create(self, model: Model, *, rngs: nnx.Rngs) -> NetworkProto:
        """Build a spin-resolved transformer wave function for the model."""
        logger.info(
            "Creating Transformers (u1u1) network: embedding_dim=%d, heads_num=%d, feed_forward_dim=%d, "
            "depth=%d, tail_hidden_dim=%d",
            self.embedding_dim,
            self.heads_num,
            self.feed_forward_dim,
            self.depth,
            self.tail_hidden_dim,
        )
        return TransformersWaveFunctionUpDown(
            double_sites=model.n_qubits,
            spin_up=(model.n_electrons + model.n_spins) // 2,
            spin_down=(model.n_electrons - model.n_spins) // 2,
            embedding_dim=self.embedding_dim,
            heads_num=self.heads_num,
            feed_forward_dim=self.feed_forward_dim,
            depth=self.depth,
            tail_hidden_dim=self.tail_hidden_dim,
            ordering=self.ordering,
            rngs=rngs,
        )


Model.network_dict["transformers/u1u1"] = TransformersUpDownConfig


@dataclasses.dataclass
class TransformersElectronConfig:
    """Transformer network with total electron-number conservation."""

    embedding_dim: int = 512
    heads_num: int = 8
    feed_forward_dim: int = 2048
    depth: int = 6
    tail_hidden_dim: int = 512
    ordering: int = 1

    def create(self, model: Model, *, rngs: nnx.Rngs) -> NetworkProto:
        """Build a total-electron transformer wave function for the model."""
        logger.info(
            "Creating Transformers (u1) network: embedding_dim=%d, heads_num=%d, feed_forward_dim=%d, "
            "depth=%d, tail_hidden_dim=%d",
            self.embedding_dim,
            self.heads_num,
            self.feed_forward_dim,
            self.depth,
            self.tail_hidden_dim,
        )
        return TransformersWaveFunctionElectron(
            sites=model.n_qubits,
            electrons=model.n_electrons,
            embedding_dim=self.embedding_dim,
            heads_num=self.heads_num,
            feed_forward_dim=self.feed_forward_dim,
            depth=self.depth,
            tail_hidden_dim=self.tail_hidden_dim,
            ordering=self.ordering,
            rngs=rngs,
        )


Model.network_dict["transformers/u1"] = TransformersElectronConfig
