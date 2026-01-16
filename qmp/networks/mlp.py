"""
This file implements the MLP network from https://arxiv.org/pdf/2109.12606 with the sampling method introduced in https://arxiv.org/pdf/2408.07625.
"""

import itertools
import torch
from ..utility.bitspack import pack_int, unpack_int


class FakeLinear(torch.nn.Module):
    """
    A fake linear layer with zero input dimension to avoid PyTorch initialization warnings.
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        assert dim_in == 0
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.zeros([dim_out]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the fake linear layer.
        """
        batch, _ = x.shape
        return self.bias.view([1, -1]).expand([batch, -1])


def select_linear_layer(dim_in: int, dim_out: int) -> torch.nn.Module:
    """
    Selects between a fake linear layer and a standard one to avoid initialization warnings when dim_in is zero.
    """
    if dim_in == 0:
        return FakeLinear(dim_in, dim_out)
    else:
        return torch.nn.Linear(dim_in, dim_out)


class MLP(torch.nn.Module):
    """
    This module implements multiple layers MLP with given dim_input, dim_output and hidden_size.
    """

    def __init__(self, dim_input: int, dim_output: int, hidden_size: tuple[int, ...]) -> None:
        super().__init__()
        self.dim_input: int = dim_input
        self.dim_output: int = dim_output
        self.hidden_size: tuple[int, ...] = hidden_size

        dimensions: list[int] = [dim_input] + list(hidden_size) + [dim_output]
        linears: list[torch.nn.Module] = [select_linear_layer(i, j) for i, j in itertools.pairwise(dimensions)]
        modules: list[torch.nn.Module] = [module for linear in linears for module in (linear, torch.nn.SiLU())][:-1]
        self.model: torch.nn.Module = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP.
        """
        return self.model(x)


class WaveFunctionElectronUpDown(torch.nn.Module):
    """
    The wave function for the MLP network.
    This module maintains the conservation of particle number for spin-up and spin-down electrons.
    """

    def __init__(
        self,
        *,
        double_sites: int,  # Number of qubits, where each pair of qubits represents a site in the MLP model
        physical_dim: int,  # Dimension of the physical space, which is always 2 for MLP
        is_complex: bool,  # Indicates whether the wave function is complex-valued, which is always true for MLP
        spin_up: int,  # Number of spin-up electrons
        spin_down: int,  # Number of spin-down electrons
        hidden_size: tuple[int, ...],  # Hidden layer sizes for the MLPs used in the amplitude and phase networks
        ordering: int
        | list[int],  # Ordering of sites: +1 for normal order, -1 for reversed order, or a custom order list
    ) -> None:
        super().__init__()
        assert double_sites % 2 == 0
        self.double_sites: int = double_sites
        self.sites: int = double_sites // 2
        assert physical_dim == 2
        assert is_complex == True  # noqa: E712
        self.spin_up: int = spin_up
        self.spin_down: int = spin_down
        self.hidden_size: tuple[int, ...] = hidden_size

        # Amplitude and Phase Networks for Each Site
        # The amplitude network accepts qubits from previous sites and outputs a vector of dimension 4,
        # representing the configuration of two qubits on the current site.
        # And the phase network accepts qubits from all sites and outputs the phase.
        self.amplitude: torch.nn.ModuleList = torch.nn.ModuleList(
            [MLP(i * 2, 4, self.hidden_size) for i in range(self.sites)]
        )
        self.phase: torch.nn.Module = MLP(self.double_sites, 1, self.hidden_size)

        # Site Ordering Configuration
        # +1 for normal order, -1 for reversed order
        if isinstance(ordering, int) and ordering == +1:
            ordering = list(range(self.sites))
        if isinstance(ordering, int) and ordering == -1:
            ordering = list(reversed(range(self.sites)))
        self.ordering: torch.Tensor
        self.register_buffer("ordering", torch.tensor(ordering, dtype=torch.int64))
        self.ordering_reversed: torch.Tensor
        self.register_buffer(
            "ordering_reversed",
            torch.scatter(
                torch.zeros(self.sites, dtype=torch.int64),
                0,
                self.ordering,
                torch.arange(self.sites, dtype=torch.int64),
            ),
        )

        # Dummy Parameter for Device and Dtype Retrieval
        # This parameter is used to infer the device and dtype of the model.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    @torch.jit.export
    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determine whether we could append spin up or spin down after incomplete configurations.
        """
        # x: batch_size * current_site * 2
        # x represents the incomplete configurations
        current_site = x.shape[1]
        # number: batch_size * 2
        # number denotes the total electron count for each incomplete configuration
        number = x.sum(dim=1)

        # up/down_electron/hole: batch_size
        # These variables store the count of electrons and holes for each incomplete configuration
        up_electron = number[:, 0]
        down_electron = number[:, 1]
        up_hole = current_site - up_electron
        down_hole = current_site - down_electron

        # add_up/down_electron/hole: batch_size
        # These variables determine whether it is possible to append an up/down electron/hole
        add_up_electron = up_electron < self.spin_up
        add_down_electron = down_electron < self.spin_down
        add_up_hole = up_hole < self.sites - self.spin_up
        add_down_hole = down_hole < self.sites - self.spin_down

        # add_up: batch_size * 2 * 1
        # add_down: batch_size * 1 * 2
        # These tensors represent the feasibility of adding up/down electrons/holes
        add_up = torch.stack([add_up_hole, add_up_electron], dim=-1).unsqueeze(-1)
        add_down = torch.stack([add_down_hole, add_down_electron], dim=-1).unsqueeze(-2)

        # add: batch_size * 2 * 2
        # add represents the logical AND of add_up and add_down, indicating the feasibility of appending specific electron/hole combinations
        # add[_, 0, 0] indicates the possibility of adding an up hole and a down hole
        # add[_, 0, 1] indicates the possibility of adding an up hole and a down electron
        # add[_, 1, 0] indicates the possibility of adding an up electron and a down hole
        # add[_, 1, 1] indicates the possibility of adding an up electron and a down electron
        add = torch.logical_and(add_up, add_down)

        return add

    @torch.jit.export
    def _normalize_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the log amplitude of incomplete configurations.
        """
        # x: batch_size * 2 * 2
        # param: batch_size
        param = (2 * x).exp().sum(dim=[-2, -1]).log() / 2
        x = x - param.unsqueeze(-1).unsqueeze(-1)
        # 1 = param = sqrt(sum(x.exp()^2)) now
        return x

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the wave function psi for the given configurations.
        """
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        batch_size: int = x.shape[0]
        # x: batch_size * sites * 2
        x = unpack_int(x, size=1, last_dim=self.double_sites).view([batch_size, self.sites, 2])
        # Apply ordering
        x = torch.index_select(x, 1, self.ordering_reversed)

        x_float: torch.Tensor = x.to(dtype=dtype)
        arange: torch.Tensor = torch.arange(batch_size, device=device)
        total_amplitude: torch.Tensor = torch.zeros([batch_size], device=device, dtype=dtype).double()
        for i, amplitude_m in enumerate(self.amplitude):
            # delta_amplitude: batch_size * 2 * 2
            # delta_amplitude represents the amplitude changes for the configurations at the new site.
            delta_amplitude: torch.Tensor = amplitude_m(x_float[:, :i].view([batch_size, 2 * i])).view(
                [batch_size, 2, 2]
            )
            # Apply a filter mask to the amplitude to ensure the conservation of particle number.
            delta_amplitude = delta_amplitude + torch.where(self._mask(x[:, :i]), 0, -torch.inf)

            # normalized_delta_amplitude: batch_size * 2 * 2
            # Normalize the delta amplitude.
            normalized_delta_amplitude: torch.Tensor = self._normalize_amplitude(delta_amplitude)

            # selected_delta_amplitude: batch
            # Select the delta amplitude for the current site.
            xi_int32: torch.Tensor = x[:, i, :].to(dtype=torch.int32)
            selected_delta_amplitude: torch.Tensor = normalized_delta_amplitude[arange, xi_int32[:, 0], xi_int32[:, 1]]

            total_amplitude = total_amplitude + selected_delta_amplitude.double()

        total_phase: torch.Tensor = self.phase(x_float.view([batch_size, self.double_sites])).view([batch_size])

        return torch.view_as_complex(torch.stack([total_amplitude, total_phase.double()], dim=-1)).exp()

    @torch.jit.export
    def generate_unique(self, batch_size: int, block_num: int = 1) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Generate configurations uniquely.
        See https://arxiv.org/pdf/2408.07625.
        """
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        # x: local_batch_size * current_site * 2
        x: torch.Tensor = torch.empty([1, 0, 2], device=device, dtype=torch.uint8)
        # (un)perturbed_log_probability : local_batch_size
        unperturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        perturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        for i, amplitude_m in enumerate(self.amplitude):
            local_batch_size: int = x.shape[0]
            x_float: torch.Tensor = x.to(dtype=dtype)

            # delta_amplitude: batch * 2 * 2
            # delta_amplitude represents the amplitude changes for the configurations at the new site.
            local_batch_size_block = local_batch_size // block_num
            remainder = local_batch_size % block_num
            delta_amplitude_block_list: list[torch.Tensor] = []
            for j in range(block_num):
                if j < remainder:
                    current_local_batch_size_block = local_batch_size_block + 1
                else:
                    current_local_batch_size_block = local_batch_size_block
                start_index = j * local_batch_size_block + min(j, remainder)
                end_index = start_index + current_local_batch_size_block
                batch_indices_block = torch.arange(start_index, end_index, device=device, dtype=torch.int64)
                delta_amplitude_block = amplitude_m(x_float.view([local_batch_size, 2 * i])[batch_indices_block]).view(
                    [current_local_batch_size_block, 2, 2]
                )
                delta_amplitude_block_list.append(delta_amplitude_block)
            delta_amplitude: torch.Tensor = torch.cat(delta_amplitude_block_list)

            # Apply a filter mask to the amplitude to ensure the conservation of particle number.
            delta_amplitude = delta_amplitude + torch.where(self._mask(x), 0, -torch.inf)

            # normalized_delta_amplitude: batch_size * 2 * 2
            # Normalize the delta amplitude.
            normalized_delta_amplitude: torch.Tensor = self._normalize_amplitude(delta_amplitude)

            # The delta unperturbed prob for all batch and 4 adds
            l: torch.Tensor = (2 * normalized_delta_amplitude).view([local_batch_size, 4])  # noqa: E741
            # and add to get the current unperturbed prob
            l = unperturbed_probability.view([local_batch_size, 1]) + l  # noqa: E741
            # Get perturbed prob by adding GUMBEL(0)
            L: torch.Tensor = l - (-torch.rand_like(l).log()).log()  # noqa: E741
            # Get max perturbed prob
            Z: torch.Tensor = L.max(dim=-1).values.view([local_batch_size, 1])
            # Evaluate the conditioned prob
            tildeL: torch.Tensor = -torch.log(
                torch.exp(-perturbed_probability.view([local_batch_size, 1])) - torch.exp(-Z) + torch.exp(-L)
            )

            # Calculate appended configurations for 4 adds
            # local_batch_size * current_site * 2 + local_batch_size * 1 * 2
            x0: torch.Tensor = torch.cat(
                [x, torch.tensor([[0, 0]], device=device, dtype=torch.uint8).expand(local_batch_size, -1, -1)], dim=1
            )
            x1: torch.Tensor = torch.cat(
                [x, torch.tensor([[0, 1]], device=device, dtype=torch.uint8).expand(local_batch_size, -1, -1)], dim=1
            )
            x2: torch.Tensor = torch.cat(
                [x, torch.tensor([[1, 0]], device=device, dtype=torch.uint8).expand(local_batch_size, -1, -1)], dim=1
            )
            x3: torch.Tensor = torch.cat(
                [x, torch.tensor([[1, 1]], device=device, dtype=torch.uint8).expand(local_batch_size, -1, -1)], dim=1
            )

            # Cat all configurations to get x : new_local_batch_size * (current_size+1) * 2
            # (un)perturbed prob : new_local_batch_size
            x = torch.cat([x0, x1, x2, x3])
            unperturbed_probability = l.permute(1, 0).reshape([4 * local_batch_size])
            perturbed_probability = tildeL.permute(1, 0).reshape([4 * local_batch_size])

            # Filter results, only use largest batch_size ones
            selected = perturbed_probability.argsort(descending=True)[:batch_size]
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]

            # If prob = 0, filter it forcibly
            selected = perturbed_probability.isfinite()
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]

        # Apply ordering
        x = torch.index_select(x, 1, self.ordering)
        # Flatten site part of x
        x = x.view([x.size(0), self.double_sites])
        # It should return configurations, amplitudes, probabilities and multiplicities.
        # But it is unique generator, so the last two fields are None
        x = pack_int(x, size=1)

        # Calculate the amplitude for the generated configurations in the batch.
        real_batch_size = len(x)
        real_batch_size_block = real_batch_size // block_num
        remainder = real_batch_size % block_num
        amplitude_list = []
        for j in range(block_num):
            if j < remainder:
                current_real_batch_size_block = real_batch_size_block + 1
            else:
                current_real_batch_size_block = real_batch_size_block
            start_index = j * real_batch_size_block + min(j, remainder)
            end_index = start_index + current_real_batch_size_block
            batch_indices_block = torch.arange(start_index, end_index, device=device, dtype=torch.int64)
            amplitude_block = self(x[batch_indices_block])
            amplitude_list.append(amplitude_block)
        amplitude: torch.Tensor = torch.cat(amplitude_list)

        return x, amplitude, None, None


class WaveFunctionElectron(torch.nn.Module):
    """
    The wave function for the MLP network.
    This module maintains the conservation of total particle number.
    """

    def __init__(
        self,
        *,
        sites: int,  # Number of qubits
        physical_dim: int,  # Dimension of the physical space, which is always 2 for MLP
        is_complex: bool,  # Indicates whether the wave function is complex-valued, which is always true for MLP
        electrons: int,  # Total number of electrons
        hidden_size: tuple[int, ...],  # Hidden layer sizes for the MLPs used in the amplitude and phase networks
        ordering: int
        | list[int],  # Ordering of sites: +1 for normal order, -1 for reversed order, or a custom order list
    ) -> None:
        super().__init__()
        self.sites: int = sites
        assert physical_dim == 2
        assert is_complex == True  # noqa: E712
        self.electrons: int = electrons
        self.hidden_size: tuple[int, ...] = hidden_size

        # Amplitude and Phase Networks for Each Site
        # The amplitude network takes in qubits from previous sites and outputs a vector of dimension 2, representing the configuration of the qubit at the current site.
        # And the phase network accepts qubits from all sites and outputs the phase.
        self.amplitude: torch.nn.ModuleList = torch.nn.ModuleList(
            [MLP(i, 2, self.hidden_size) for i in range(self.sites)]
        )
        self.phase: torch.nn.Module = MLP(self.sites, 1, self.hidden_size)

        # Site Ordering Configuration
        # +1 for normal order, -1 for reversed order
        if isinstance(ordering, int) and ordering == +1:
            ordering = list(range(self.sites))
        if isinstance(ordering, int) and ordering == -1:
            ordering = list(reversed(range(self.sites)))
        self.ordering: torch.Tensor
        self.register_buffer("ordering", torch.tensor(ordering, dtype=torch.int64))
        self.ordering_reversed: torch.Tensor
        self.register_buffer(
            "ordering_reversed",
            torch.scatter(
                torch.zeros(self.sites, dtype=torch.int64),
                0,
                self.ordering,
                torch.arange(self.sites, dtype=torch.int64),
            ),
        )

        # Dummy Parameter for Device and Dtype Retrieval
        # This parameter is used to infer the device and dtype of the model.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    @torch.jit.export
    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determine whether we could append an electron or a hole after incomplete configurations.
        """
        # x: batch_size * current_site
        current_site = x.shape[1]
        # number: batch_size
        number = x.sum(dim=1)

        electron = number
        hole = current_site - electron

        # add_electron/hole: batch_size
        add_electron = electron < self.electrons
        add_hole = hole < self.sites - self.electrons

        # return: batch_size * 2
        return torch.stack([add_hole, add_electron], dim=-1)

    @torch.jit.export
    def _normalize_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the log amplitude of incomplete configurations.
        """
        # x: batch_size * 2
        # param: batch_size
        param = (2 * x).exp().sum(dim=[-1]).log() / 2
        x = x - param.unsqueeze(-1)
        # 1 = param = sqrt(sum(x.exp()^2)) now
        return x

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the wave function psi for the given configurations.
        """
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        batch_size: int = x.shape[0]
        # x: batch_size * sites
        x = unpack_int(x, size=1, last_dim=self.sites).view([batch_size, self.sites])
        # Apply ordering
        x = torch.index_select(x, 1, self.ordering_reversed)

        x_float: torch.Tensor = x.to(dtype=dtype)
        arange: torch.Tensor = torch.arange(batch_size, device=device)
        total_amplitude: torch.Tensor = torch.zeros([batch_size], device=device, dtype=dtype).double()
        for i, amplitude_m in enumerate(self.amplitude):
            # delta_amplitude : batch_size * 2
            # delta_amplitude represents the amplitude changes for the configurations at the new site.
            delta_amplitude: torch.Tensor = amplitude_m(x_float[:, :i].view([batch_size, i])).view([batch_size, 2])
            # Apply a filter mask to the amplitude to ensure the conservation of particle number.
            delta_amplitude = delta_amplitude + torch.where(self._mask(x[:, :i]), 0, -torch.inf)

            # normalized_delta_amplitude: batch_size * 2
            # Normalize the delta amplitude.
            normalized_delta_amplitude: torch.Tensor = self._normalize_amplitude(delta_amplitude)

            # selected_delta_amplitude: batch
            # Select the delta amplitude for the current site.
            xi_int32: torch.Tensor = x[:, i].to(dtype=torch.int32)
            selected_delta_amplitude: torch.Tensor = normalized_delta_amplitude[arange, xi_int32]

            total_amplitude = total_amplitude + selected_delta_amplitude.double()

        total_phase: torch.Tensor = self.phase(x_float.view([batch_size, self.sites])).view([batch_size])

        return torch.view_as_complex(torch.stack([total_amplitude, total_phase.double()], dim=-1)).exp()

    @torch.jit.export
    def generate_unique(self, batch_size: int, block_num: int = 1) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Generate configurations uniquely.
        See https://arxiv.org/pdf/2408.07625.
        """
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        # x: local_batch_size * current_site
        x: torch.Tensor = torch.empty([1, 0], device=device, dtype=torch.uint8)
        # (un)perturbed_log_probability : local_batch_size
        unperturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        perturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        for i, amplitude_m in enumerate(self.amplitude):
            local_batch_size: int = x.shape[0]
            x_float: torch.Tensor = x.to(dtype=dtype)

            # delta_amplitude: batch * 2
            # delta_amplitude represents the amplitude changes for the configurations at the new site.
            local_batch_size_block = local_batch_size // block_num
            remainder = local_batch_size % block_num
            delta_amplitude_block_list: list[torch.Tensor] = []
            for j in range(block_num):
                if j < remainder:
                    current_local_batch_size_block = local_batch_size_block + 1
                else:
                    current_local_batch_size_block = local_batch_size_block
                start_index = j * local_batch_size_block + min(j, remainder)
                end_index = start_index + current_local_batch_size_block
                batch_indices_block = torch.arange(start_index, end_index, device=device, dtype=torch.int64)
                delta_amplitude_block = amplitude_m(x_float.view([local_batch_size, i])[batch_indices_block]).view(
                    [current_local_batch_size_block, 2]
                )
                delta_amplitude_block_list.append(delta_amplitude_block)
            delta_amplitude: torch.Tensor = torch.cat(delta_amplitude_block_list)

            # Apply a filter mask to the amplitude to ensure the conservation of particle number.
            delta_amplitude = delta_amplitude + torch.where(self._mask(x), 0, -torch.inf)

            # normalized_delta_amplitude: batch_size * 2
            # Normalize the delta amplitude.
            normalized_delta_amplitude: torch.Tensor = self._normalize_amplitude(delta_amplitude)

            # The delta unperturbed prob for all batch and 2 adds
            l: torch.Tensor = (2 * normalized_delta_amplitude).view([local_batch_size, 2])  # noqa: E741
            # and add to get the current unperturbed prob
            l = unperturbed_probability.view([local_batch_size, 1]) + l  # noqa: E741
            # Get perturbed prob by adding GUMBEL(0)
            L: torch.Tensor = l - (-torch.rand_like(l).log()).log()  # noqa: E741
            # Get max perturbed prob
            Z: torch.Tensor = L.max(dim=-1).values.view([local_batch_size, 1])
            # Evaluate the conditioned prob
            tildeL: torch.Tensor = -torch.log(
                torch.exp(-perturbed_probability.view([local_batch_size, 1])) - torch.exp(-Z) + torch.exp(-L)
            )

            # Calculate appended configurations for 2 adds
            # local_batch_size * current_site + local_batch_size * 1
            x0: torch.Tensor = torch.cat(
                [x, torch.tensor([0], device=device, dtype=torch.uint8).expand(local_batch_size, -1)], dim=1
            )
            x1: torch.Tensor = torch.cat(
                [x, torch.tensor([1], device=device, dtype=torch.uint8).expand(local_batch_size, -1)], dim=1
            )

            # Cat all configurations to get x : new_local_batch_size * (current_size+1) * 2
            # (un)perturbed prob : new_local_batch_size
            x = torch.cat([x0, x1])
            unperturbed_probability = l.permute(1, 0).reshape([2 * local_batch_size])
            perturbed_probability = tildeL.permute(1, 0).reshape([2 * local_batch_size])

            # Filter results, only use largest batch_size ones
            selected = perturbed_probability.argsort(descending=True)[:batch_size]
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]

            # If prob = 0, filter it forcibly
            selected = perturbed_probability.isfinite()
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]

        # Apply ordering
        x = torch.index_select(x, 1, self.ordering)
        # Flatten site part of x
        x = x.view([x.size(0), self.sites])
        # It should return configurations, amplitudes, probabilities and multiplicities.
        # But it is unique generator, so the last two fields are None
        x = pack_int(x, size=1)

        # Calculate the amplitude for the generated configurations in the batch.
        real_batch_size = len(x)
        real_batch_size_block = real_batch_size // block_num
        remainder = real_batch_size % block_num
        amplitude_list = []
        for j in range(block_num):
            if j < remainder:
                current_real_batch_size_block = real_batch_size_block + 1
            else:
                current_real_batch_size_block = real_batch_size_block
            start_index = j * real_batch_size_block + min(j, remainder)
            end_index = start_index + current_real_batch_size_block
            batch_indices_block = torch.arange(start_index, end_index, device=device, dtype=torch.int64)
            amplitude_block = self(x[batch_indices_block])
            amplitude_list.append(amplitude_block)
        amplitude: torch.Tensor = torch.cat(amplitude_list)

        return x, amplitude, None, None


class WaveFunctionNormal(torch.nn.Module):
    """
    The wave function for the MLP network.
    This module does not maintain any conservation.
    """

    def __init__(
        self,
        *,
        sites: int,  # Number of qubits
        physical_dim: int,  # Dimension of the physical space, which is always 2 for MLP
        is_complex: bool,  # Indicates whether the wave function is complex-valued, which is always true for MLP
        hidden_size: tuple[int, ...],  # Hidden layer sizes for the MLPs used in the amplitude and phase networks
        ordering: int
        | list[int],  # Ordering of sites: +1 for normal order, -1 for reversed order, or a custom order list
    ) -> None:
        super().__init__()
        self.sites: int = sites
        assert physical_dim == 2
        assert is_complex == True  # noqa: E712
        self.hidden_size: tuple[int, ...] = hidden_size

        # Amplitude and Phase Networks for Each Site
        # The amplitude network takes in qubits from previous sites and outputs a vector of dimension 2, representing the configuration of the qubit at the current site.
        # And the phase network accepts qubits from all sites and outputs the phase.
        self.amplitude: torch.nn.ModuleList = torch.nn.ModuleList(
            [MLP(i, 2, self.hidden_size) for i in range(self.sites)]
        )
        self.phase: torch.nn.Module = MLP(self.sites, 1, self.hidden_size)

        # Site Ordering Configuration
        # +1 for normal order, -1 for reversed order
        if isinstance(ordering, int) and ordering == +1:
            ordering = list(range(self.sites))
        if isinstance(ordering, int) and ordering == -1:
            ordering = list(reversed(range(self.sites)))
        self.ordering: torch.Tensor
        self.register_buffer("ordering", torch.tensor(ordering, dtype=torch.int64))
        self.ordering_reversed: torch.Tensor
        self.register_buffer(
            "ordering_reversed",
            torch.scatter(
                torch.zeros(self.sites, dtype=torch.int64),
                0,
                self.ordering,
                torch.arange(self.sites, dtype=torch.int64),
            ),
        )

        # Dummy Parameter for Device and Dtype Retrieval
        # This parameter is used to infer the device and dtype of the model.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    @torch.jit.export
    def _normalize_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the log amplitude of incomplete configurations.
        """
        # x: batch_size * 2
        # param:  batch_size
        param = (2 * x).exp().sum(dim=[-1]).log() / 2
        x = x - param.unsqueeze(-1)
        # 1 = param = sqrt(sum(x.exp()^2)) now
        return x

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the wave function psi for the given configurations.
        """
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        batch_size: int = x.shape[0]
        # x: batch_size * sites
        x = unpack_int(x, size=1, last_dim=self.sites).view([batch_size, self.sites])
        # Apply ordering
        x = torch.index_select(x, 1, self.ordering_reversed)

        x_float: torch.Tensor = x.to(dtype=dtype)
        arange: torch.Tensor = torch.arange(batch_size, device=device)
        total_amplitude: torch.Tensor = torch.zeros([batch_size], device=device, dtype=dtype).double()
        for i, amplitude_m in enumerate(self.amplitude):
            # delta_amplitude : batch_size * 2
            # delta_amplitude represents the amplitude changes for the configurations at the new site.
            delta_amplitude: torch.Tensor = amplitude_m(x_float[:, :i].view([batch_size, i])).view([batch_size, 2])

            # normalized_delta_amplitude: batch_size * 2
            # Normalize the delta amplitude.
            normalized_delta_amplitude: torch.Tensor = self._normalize_amplitude(delta_amplitude)

            # selected_delta_amplitude: batch
            # Select the delta amplitude for the current site.
            xi_int32: torch.Tensor = x[:, i].to(dtype=torch.int32)
            selected_delta_amplitude: torch.Tensor = normalized_delta_amplitude[arange, xi_int32]

            total_amplitude = total_amplitude + selected_delta_amplitude.double()

        total_phase: torch.Tensor = self.phase(x_float.view([batch_size, self.sites])).view([batch_size])

        return torch.view_as_complex(torch.stack([total_amplitude, total_phase.double()], dim=-1)).exp()

    @torch.jit.export
    def generate_unique(self, batch_size: int, block_num: int = 1) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Generate configurations uniquely.
        See https://arxiv.org/pdf/2408.07625.
        """
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        # x: local_batch_size * current_site
        x: torch.Tensor = torch.empty([1, 0], device=device, dtype=torch.uint8)
        # (un)perturbed_log_probability : local_batch_size
        unperturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        perturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        for i, amplitude_m in enumerate(self.amplitude):
            local_batch_size: int = x.shape[0]
            x_float: torch.Tensor = x.to(dtype=dtype)

            # delta_amplitude: batch * 2
            # delta_amplitude represents the amplitude changes for the configurations at the new site.
            local_batch_size_block = local_batch_size // block_num
            remainder = local_batch_size % block_num
            delta_amplitude_block_list: list[torch.Tensor] = []
            for j in range(block_num):
                if j < remainder:
                    current_local_batch_size_block = local_batch_size_block + 1
                else:
                    current_local_batch_size_block = local_batch_size_block
                start_index = j * local_batch_size_block + min(j, remainder)
                end_index = start_index + current_local_batch_size_block
                batch_indices_block = torch.arange(start_index, end_index, device=device, dtype=torch.int64)
                delta_amplitude_block = amplitude_m(x_float.view([local_batch_size, i])[batch_indices_block]).view(
                    [current_local_batch_size_block, 2]
                )
                delta_amplitude_block_list.append(delta_amplitude_block)
            delta_amplitude: torch.Tensor = torch.cat(delta_amplitude_block_list)

            # normalized_delta_amplitude: batch_size * 2
            # Normalize the delta amplitude.
            normalized_delta_amplitude: torch.Tensor = self._normalize_amplitude(delta_amplitude)

            # The delta unperturbed prob for all batch and 2 adds
            l: torch.Tensor = (2 * normalized_delta_amplitude).view([local_batch_size, 2])  # noqa: E741
            # and add to get the current unperturbed prob
            l = unperturbed_probability.view([local_batch_size, 1]) + l  # noqa: E741
            # Get perturbed prob by adding GUMBEL(0)
            L: torch.Tensor = l - (-torch.rand_like(l).log()).log()  # noqa: E741
            # Get max perturbed prob
            Z: torch.Tensor = L.max(dim=-1).values.view([local_batch_size, 1])
            # Evaluate the conditioned prob
            tildeL: torch.Tensor = -torch.log(
                torch.exp(-perturbed_probability.view([local_batch_size, 1])) - torch.exp(-Z) + torch.exp(-L)
            )

            # Calculate appended configurations for 2 adds
            # local_batch_size * current_site + local_batch_size * 1
            x0: torch.Tensor = torch.cat(
                [x, torch.tensor([0], device=device, dtype=torch.uint8).expand(local_batch_size, -1)], dim=1
            )
            x1: torch.Tensor = torch.cat(
                [x, torch.tensor([1], device=device, dtype=torch.uint8).expand(local_batch_size, -1)], dim=1
            )

            # Cat all configurations to get x : new_local_batch_size * (current_size+1) * 2
            # (un)perturbed prob : new_local_batch_size
            x = torch.cat([x0, x1])
            unperturbed_probability = l.permute(1, 0).reshape([2 * local_batch_size])
            perturbed_probability = tildeL.permute(1, 0).reshape([2 * local_batch_size])

            # Filter results, only use largest batch_size ones
            selected = perturbed_probability.argsort(descending=True)[:batch_size]
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]

            # If prob = 0, filter it forcibly
            selected = perturbed_probability.isfinite()
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]

        # Apply ordering
        x = torch.index_select(x, 1, self.ordering)
        # Flatten site part of x
        x = x.view([x.size(0), self.sites])
        # It should return configurations, amplitudes, probabilities and multiplicities.
        # But it is unique generator, so the last two fields are None
        x = pack_int(x, size=1)

        # Calculate the amplitude for the generated configurations in the batch.
        real_batch_size = len(x)
        real_batch_size_block = real_batch_size // block_num
        remainder = real_batch_size % block_num
        amplitude_list = []
        for j in range(block_num):
            if j < remainder:
                current_real_batch_size_block = real_batch_size_block + 1
            else:
                current_real_batch_size_block = real_batch_size_block
            start_index = j * real_batch_size_block + min(j, remainder)
            end_index = start_index + current_real_batch_size_block
            batch_indices_block = torch.arange(start_index, end_index, device=device, dtype=torch.int64)
            amplitude_block = self(x[batch_indices_block])
            amplitude_list.append(amplitude_block)
        amplitude: torch.Tensor = torch.cat(amplitude_list)

        return x, amplitude, None, None
