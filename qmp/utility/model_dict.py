"""
This module is used to store a dictionary that maps model names to their corresponding models.

Other packages or subpackages can register their models by adding entries to this dictionary.
"""

import typing
import torch


class NetworkProto(typing.Protocol):
    """
    The Network protocol defines the interface that all networks must implement.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the amplitude for the given configurations.

        Parameters
        ----------
        x : torch.Tensor
            The configurations to calculate the amplitude for.
            The configurations are a two-dimensional uint8 tensor with first dimension equal to some batch size.
            The second dimension contains occupation for each qubit which is bitwise encoded.

        Returns
        -------
        torch.Tensor
            The amplitudes of the configurations.
            The amplitudes are a one-dimensional complex tensor with the only dimension equal to the batch size.
        """

    def generate_unique(self, batch_size: int, block_num: int = 1) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Generate a batch of unique configurations.

        Parameters
        ----------
        batch_size : int
            The number of configurations to generate.
        block_num : int, default=1
            The number of batch block to generate. It is used to split the batch into smaller parts to avoid memory issues.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, None, None]
            A tuple containing the generated configurations, their amplitudes, and two None values.
            The configurations are a two-dimensional uint8 tensor with first dimension equal to `batch_size`.
            The second dimension contains occupation for each qubit which is bitwise encoded.
            The amplitudes are a one-dimensional complex tensor with the only dimension equal to `batch_size`.
            The last two None values are reserved for future use.
        """

    def load_state_dict(self, data: dict[str, torch.Tensor]) -> typing.Any:
        """torch.nn.Module function"""

    def state_dict(self) -> dict[str, torch.Tensor]:
        """torch.nn.Module function"""

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> typing.Any:
        """torch.nn.Module function"""

    def parameters(self) -> typing.Iterable[torch.Tensor]:
        """torch.nn.Module function"""

    def bfloat16(self) -> typing.Self:
        """torch.nn.Module function"""

    def half(self) -> typing.Self:
        """torch.nn.Module function"""

    def float(self) -> typing.Self:
        """torch.nn.Module function"""

    def double(self) -> typing.Self:
        """torch.nn.Module function"""


Model_contra = typing.TypeVar("Model_contra", contravariant=True)


class NetworkConfigProto(typing.Protocol[Model_contra]):
    """
    The NetworkConfigProto protocol defines the interface that all network configs must implement.
    """

    def create(self, model: Model_contra) -> NetworkProto:
        """
        Create the network from the given config for the given model.
        """


ModelConfig = typing.TypeVar("ModelConfig")


class ModelProto(typing.Protocol[ModelConfig]):
    """
    The Model protocol defines the interface that all models must implement.
    """

    network_dict: dict[str, type[NetworkConfigProto[typing.Self]]]

    ref_energy: float

    config_t: type[ModelConfig]

    def __init__(self, config: ModelConfig) -> None:
        """
        Create a model from the given config.

        Parameters
        ----------
        config : ModelConfig
            The config of model.
        """

    def apply_within(self, configs_i: torch.Tensor, psi_i: torch.Tensor, configs_j: torch.Tensor) -> torch.Tensor:
        """
        Applies the Hamiltonian to the given vector.

        Parameters
        ----------
        configs_i : torch.Tensor
            The configurations to apply the Hamiltonian to.
        psi_i : torch.Tensor
            The amplitudes of the configurations.
        configs_j : torch.Tensor
            The configurations subspace for the result of the Hamiltonian application.

        Returns
        -------
        torch.Tensor
            The result of the Hamiltonian application on the selected configurations subspace.
        """

    def find_relative(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        count_selected: int,
        configs_exclude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Find relative configurations to the given configurations.

        Parameters
        ----------
        configs_i : torch.Tensor
            The configurations to find relative configurations for.
        psi_i : torch.Tensor
            The amplitudes of the configurations.
        count_selected : int
            The number of relative configurations to find.
        configs_exclude : torch.Tensor, optional
            The configurations to exclude from the result.

        Returns
        -------
        torch.Tensor
            The relative configurations.
        """

    def list_relative(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        configs_exclude: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        List all unique relative configurations and their accumulated amplitudes.

        Parameters
        ----------
        configs_i : torch.Tensor
            Input configurations (uint8).
        psi_i : torch.Tensor
            Input amplitudes (complex64).
        configs_exclude : torch.Tensor, optional
            Configurations to exclude from the result. Defaults to configs_i.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (configs_j, psi_j) where configs_j are unique new configurations
            and psi_j are their summed amplitudes from all connected paths.
        """

    def diagonal_term(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Calculate the diagonal term for the given configurations.

        Parameters
        ----------
        configs : torch.Tensor
            The configurations to calculate the diagonal term for.

        Returns
        -------
        torch.Tensor
            The diagonal term of the configurations.
        """

    def show_config(self, config: torch.Tensor) -> str:
        """
        Converts a configuration tensor to a string representation.

        Parameters
        ----------
        config : torch.Tensor
            The configuration tensor to convert.

        Returns
        -------
        str
            The string representation of the configuration tensor.
        """


model_dict: dict[str, typing.Type[ModelProto]] = {}
