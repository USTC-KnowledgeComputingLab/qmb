"""
This file contains the Hamiltonian class, which is used to store the Hamiltonian and process iteration over each term in the Hamiltonian for given configurations.
"""

import os
import logging
import platformdirs
import typing
import dataclasses
import torch
import torch.utils.cpp_extension
import torch.distributed.rpc as rpc

from ..utility.distributed import get_rank, get_world_size, get_local_device, is_rank_zero


class Hamiltonian:
    """
    The Hamiltonian type, which stores the Hamiltonian and processes iteration over each term in the Hamiltonian for given configurations.

    Supports distributed computation via torch.distributed.rpc. When world_size > 1,
    computation is distributed across all ranks with results aggregated on rank 0.
    """

    _hamiltonian_module: dict[tuple[str, int, int], object] = {}

    @classmethod
    def _set_torch_cuda_arch_list(cls) -> None:
        """
        Set the CUDA architecture list for PyTorch to use when compiling the PyTorch extensions.
        """
        if not torch.cuda.is_available():
            return
        if "TORCH_CUDA_ARCH_LIST" in os.environ:
            return
        os.environ["TORCH_CUDA_ARCH_LIST"] = "native"

    @classmethod
    def _load_module(cls, device_type: str = "declaration", n_qubytes: int = 0, particle_cut: int = 0) -> object:
        """
        Load the Hamiltonian PyTorch extension module for the given device type, number of qubytes, and particle cut or just load the declaration module.
        """
        cls._set_torch_cuda_arch_list()
        if device_type != "declaration":
            cls._load_module("declaration", n_qubytes, particle_cut)  # Ensure the declaration module is loaded first
        key = (device_type, n_qubytes, particle_cut)
        is_declaration = key == ("declaration", 0, 0)
        name = "qmp_hamiltonian" if is_declaration else f"qmp_hamiltonian_{n_qubytes}_{particle_cut}"
        if key not in cls._hamiltonian_module:
            build_directory = platformdirs.user_cache_path("qmp", "kclab") / name / device_type
            build_directory.mkdir(parents=True, exist_ok=True)
            folder = os.path.dirname(__file__)
            match device_type:
                case "declaration":
                    sources = [f"{folder}/_hamiltonian.cpp"]
                case "cpu":
                    sources = [f"{folder}/_hamiltonian_cpu.cpp"]
                case "cuda":
                    sources = [f"{folder}/_hamiltonian_cuda.cu"]
                case _:
                    raise ValueError("Unsupported device type")
            cls._hamiltonian_module[key] = torch.utils.cpp_extension.load(
                name=name,
                sources=sources,
                is_python_module=is_declaration,
                extra_cflags=[
                    "-O3",
                    "-ffast-math",
                    "-march=native",
                    f"-DN_QUBYTES={n_qubytes}",
                    f"-DPARTICLE_CUT={particle_cut}",
                    "-std=c++20",
                ],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    f"-DN_QUBYTES={n_qubytes}",
                    f"-DPARTICLE_CUT={particle_cut}",
                    "-std=c++20",
                ],
                build_directory=build_directory,
            )
        if is_declaration:
            return cls._hamiltonian_module[key]
        else:
            return getattr(torch.ops, name)

    @classmethod
    def _prepare(
        cls, hamiltonian: dict[tuple[tuple[int, int], ...], complex]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse the Hamiltonian dictionary into site, kind, and coefficient tensors.
        """
        return getattr(cls._load_module(), "prepare")(hamiltonian)

    def __init__(
        self,
        hamiltonian: dict[tuple[tuple[int, int], ...], complex] | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        kind: str,
    ) -> None:
        """
        Initialize the Hamiltonian object, either from a dictionary or from pre-parsed tensors.
        """
        self.site: torch.Tensor
        self.kind: torch.Tensor
        self.coef: torch.Tensor
        if isinstance(hamiltonian, dict):
            self.site, self.kind, self.coef = self._prepare(hamiltonian)
            self._sort_site_kind_coef()
        else:
            self.site, self.kind, self.coef = hamiltonian
        self.particle_cut: int
        match kind:
            case "fermi":
                self.particle_cut = 1
            case "bose2":
                self.particle_cut = 2
            case _:
                raise ValueError(f"Unknown kind: {kind}")

    def _sort_site_kind_coef(self) -> None:
        """
        Reorder the site, kind, and coefficient tensors in descending order of the norm of the coefficients.
        """
        order = self.coef.norm(dim=1).argsort(descending=True)
        self.site = self.site[order]
        self.kind = self.kind[order]
        self.coef = self.coef[order]

    def _prepare_data(self, device: torch.device) -> None:
        """
        Prepare the site, kind, and coefficient tensors for computation on the given device.
        """
        self.site = self.site.to(device=device).contiguous()
        self.kind = self.kind.to(device=device).contiguous()
        self.coef = self.coef.to(device=device).contiguous()

    def _get_data_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get Hamiltonian data as a tuple for serialization (RPC transfer).
        """
        return (self.site.cpu(), self.kind.cpu(), self.coef.cpu(), self.particle_cut)

    def apply_within(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        configs_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies the Hamiltonian to the given vector.

        Parameters
        ----------
        configs_i : torch.Tensor
            A uint8 tensor of shape [batch_size_i, n_qubytes] representing the input configurations.
        psi_i : torch.Tensor
            A complex64 tensor of shape [batch_size_i] representing the input amplitudes on the given configurations.
        configs_j : torch.Tensor
            A uint8 tensor of shape [batch_size_j, n_qubytes] representing the output configurations.

        Returns
        -------
        torch.Tensor
            A tensor of shape [batch_size_j] representing the output amplitudes on the given configurations.
        """
        world_size = get_world_size()
        rank = get_rank()
        device = get_local_device()

        if world_size == 1:
            # Single process: compute locally
            return self._apply_within_local(configs_i, psi_i, configs_j, device)


        # Distributed: split and compute across ranks
        batch_size_i = configs_i.size(0)
        chunk_size = batch_size_i // world_size
        remainder = batch_size_i % world_size

        # Compute chunk indices for this rank
        start_idx = rank * chunk_size + min(rank, remainder)
        end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)

        if start_idx >= end_idx:
            # Empty chunk for this rank
            local_result = torch.zeros(configs_j.size(0), dtype=torch.complex64, device=device)
        else:
            configs_i_chunk = configs_i[start_idx:end_idx]
            psi_i_chunk = psi_i[start_idx:end_idx]
            local_result = self._apply_within_local(configs_i_chunk, psi_i_chunk, configs_j, device)

        # Gather results from all ranks to rank 0
        if rank == 0:

            # Send RPC requests to all other ranks FIRST (asynchronous)
            rpc_refs = []
            for target_rank in range(1, world_size):
                rpc_ref = rpc.remote(
                    f"rank_{target_rank}",
                    _remote_apply_within,
                    args=(
                        self._get_data_tuple(),
                        configs_i.to("cpu"),
                        psi_i.to("cpu"),
                        configs_j.to("cpu"),
                        target_rank,
                        batch_size_i,
                        world_size,
                    ),
                )
                rpc_refs.append(rpc_ref)


            # Now compute local chunk while RPC is running remotely
            if start_idx >= end_idx:
                local_result = torch.zeros(configs_j.size(0), dtype=torch.complex64, device=device)
            else:
                configs_i_chunk = configs_i[start_idx:end_idx]
                psi_i_chunk = psi_i[start_idx:end_idx]
                local_result = self._apply_within_local(configs_i_chunk, psi_i_chunk, configs_j, device)


            # Collect results from all RPC calls
            results = [local_result]
            for rpc_ref in rpc_refs:
                results.append(rpc_ref.to_here().to(device))

            # Sum all results
            final_result = torch.zeros(configs_j.size(0), dtype=torch.complex64, device=device)
            for r in results:
                final_result = final_result + r
            return final_result
        else:
            # Non-rank-0 workers just return local result (rank 0 will collect)
            return local_result

    def _apply_within_local(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        configs_j: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Local apply_within computation on a specific device.
        """
        configs_i = configs_i.to(device=device)
        psi_i = psi_i.to(device=device)
        configs_j = configs_j.to(device=device)
        self._prepare_data(device)

        _apply_within = getattr(
            self._load_module(device.type, configs_i.size(1), self.particle_cut),
            "apply_within",
        )
        psi_j = torch.view_as_complex(
            _apply_within(configs_i, torch.view_as_real(psi_i), configs_j, self.site, self.kind, self.coef)
        )
        return psi_j

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
            A uint8 tensor of shape [batch_size, n_qubytes] representing the input configurations.
        psi_i : torch.Tensor
            A complex64 tensor of shape [batch_size] representing the input amplitudes on the given configurations.
        count_selected : int
            The number of selected configurations to be returned.
        configs_exclude : torch.Tensor, optional
            A uint8 tensor of shape [batch_size_exclude, n_qubytes] representing the configurations to be excluded from the result, by default None

        Returns
        -------
        torch.Tensor
            The resulting configurations after applying the Hamiltonian, only the first `count_selected` configurations are guaranteed to be returned.
            The order of the configurations is guaranteed to be sorted by estimated psi for the remaining configurations.
        """
        if configs_exclude is None:
            configs_exclude = configs_i

        world_size = get_world_size()
        rank = get_rank()
        device = get_local_device()

        if world_size == 1:
            return self._find_relative_local(configs_i, psi_i, count_selected, configs_exclude, device)

        # Distributed computation
        batch_size_i = configs_i.size(0)
        chunk_size = batch_size_i // world_size
        remainder = batch_size_i % world_size

        start_idx = rank * chunk_size + min(rank, remainder)
        end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)

        if start_idx >= end_idx:
            local_configs = torch.empty(0, configs_i.size(1), dtype=configs_i.dtype, device=device)
        else:
            configs_i_chunk = configs_i[start_idx:end_idx]
            psi_i_chunk = psi_i[start_idx:end_idx]
            local_configs = self._find_relative_local(configs_i_chunk, psi_i_chunk, count_selected, configs_exclude, device)

        if rank == 0:
            # Send RPC requests to all other ranks FIRST (asynchronous)
            rpc_refs = []
            for target_rank in range(1, world_size):
                rpc_ref = rpc.remote(
                    f"rank_{target_rank}",
                    _remote_find_relative,
                    args=(
                        self._get_data_tuple(),
                        configs_i.to("cpu"),
                        psi_i.to("cpu"),
                        configs_exclude.to("cpu"),
                        target_rank,
                        batch_size_i,
                        world_size,
                        count_selected,
                    ),
                )
                rpc_refs.append(rpc_ref)

            # Now compute local chunk while RPC is running remotely
            if start_idx >= end_idx:
                local_configs = torch.empty(0, configs_i.size(1), dtype=configs_i.dtype, device=device)
            else:
                configs_i_chunk = configs_i[start_idx:end_idx]
                psi_i_chunk = psi_i[start_idx:end_idx]
                local_configs = self._find_relative_local(configs_i_chunk, psi_i_chunk, count_selected, configs_exclude, device)

            # Collect results from all RPC calls
            results = [local_configs]
            for rpc_ref in rpc_refs:
                results.append(rpc_ref.to_here().to(device))

            # Merge and deduplicate
            if len(results) == 0 or all(r.size(0) == 0 for r in results):
                return torch.empty(0, configs_i.size(1), dtype=configs_i.dtype, device=device)

            all_configs = torch.cat([r for r in results if r.size(0) > 0], dim=0)
            unique_configs = torch.unique(all_configs, sorted=True, dim=0)

            # Exclude configs_exclude
            configs_exclude_dev = configs_exclude.to(device)
            exclude_mask = (unique_configs.unsqueeze(1) == configs_exclude_dev.unsqueeze(0)).all(dim=-1).any(dim=-1)
            filtered_configs = unique_configs[~exclude_mask]

            if filtered_configs.size(0) > count_selected:
                filtered_configs = filtered_configs[:count_selected]
            return filtered_configs
        else:
            return local_configs

    def _find_relative_local(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        count_selected: int,
        configs_exclude: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Local find_relative computation on a specific device.
        """
        configs_i = configs_i.to(device=device)
        psi_i = psi_i.to(device=device)
        configs_exclude = configs_exclude.to(device=device)
        self._prepare_data(device)

        _find_relative = getattr(
            self._load_module(device.type, configs_i.size(1), self.particle_cut),
            "find_relative",
        )
        configs_j = _find_relative(
            configs_i, torch.view_as_real(psi_i), count_selected, self.site, self.kind, self.coef, configs_exclude
        )
        return configs_j

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
        if configs_exclude is None:
            configs_exclude = configs_i

        world_size = get_world_size()
        rank = get_rank()
        device = get_local_device()

        if world_size == 1:
            return self._list_relative_local(configs_i, psi_i, configs_exclude, device)

        # Distributed computation
        batch_size_i = configs_i.size(0)
        chunk_size = batch_size_i // world_size
        remainder = batch_size_i % world_size

        start_idx = rank * chunk_size + min(rank, remainder)
        end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)

        if start_idx >= end_idx:
            local_configs = torch.empty(0, configs_i.size(1), dtype=configs_i.dtype, device=device)
            local_psi = torch.empty(0, dtype=psi_i.dtype, device=device)
        else:
            configs_i_chunk = configs_i[start_idx:end_idx]
            psi_i_chunk = psi_i[start_idx:end_idx]
            local_configs, local_psi = self._list_relative_local(configs_i_chunk, psi_i_chunk, configs_exclude, device)

        if rank == 0:
            # Send RPC requests to all other ranks FIRST (asynchronous)
            rpc_refs = []
            for target_rank in range(1, world_size):
                rpc_ref = rpc.remote(
                    f"rank_{target_rank}",
                    _remote_list_relative,
                    args=(
                        self._get_data_tuple(),
                        configs_i.to("cpu"),
                        psi_i.to("cpu"),
                        configs_exclude.to("cpu"),
                        target_rank,
                        batch_size_i,
                        world_size,
                    ),
                )
                rpc_refs.append(rpc_ref)

            # Now compute local chunk while RPC is running remotely
            if start_idx >= end_idx:
                local_configs = torch.empty(0, configs_i.size(1), dtype=configs_i.dtype, device=device)
                local_psi = torch.empty(0, dtype=psi_i.dtype, device=device)
            else:
                configs_i_chunk = configs_i[start_idx:end_idx]
                psi_i_chunk = psi_i[start_idx:end_idx]
                local_configs, local_psi = self._list_relative_local(configs_i_chunk, psi_i_chunk, configs_exclude, device)

            # Collect results from all RPC calls
            results_configs = [local_configs]
            results_psi = [local_psi]
            for rpc_ref in rpc_refs:
                remote_configs, remote_psi = rpc_ref.to_here()
                results_configs.append(remote_configs.to(device))
                results_psi.append(remote_psi.to(device))

            # Handle empty results
            if len(results_configs) == 0 or all(r.size(0) == 0 for r in results_configs):
                return torch.empty(0, configs_i.size(1), dtype=configs_i.dtype, device=device), \
                       torch.empty(0, dtype=psi_i.dtype, device=device)

            # Merge and deduplicate
            all_configs = torch.cat([r for r in results_configs if r.size(0) > 0], dim=0)
            all_psi = torch.cat([r for r in results_psi if r.size(0) > 0], dim=0)

            unique_configs, inverse_indices = torch.unique(all_configs, return_inverse=True, dim=0)
            unique_psi = torch.zeros(unique_configs.size(0), dtype=all_psi.dtype, device=device)
            unique_psi.scatter_add_(0, inverse_indices, all_psi)

            # Exclude configs_exclude
            configs_exclude_dev = configs_exclude.to(device)
            exclude_mask = (unique_configs.unsqueeze(1) == configs_exclude_dev.unsqueeze(0)).all(dim=-1).any(dim=-1)
            filtered_configs = unique_configs[~exclude_mask]
            filtered_psi = unique_psi[~exclude_mask]

            return filtered_configs, filtered_psi
        else:
            return local_configs, local_psi

    def _list_relative_local(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        configs_exclude: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Local list_relative computation on a specific device.
        """
        configs_i = configs_i.to(device=device)
        psi_i = psi_i.to(device=device)
        configs_exclude = configs_exclude.to(device=device)
        self._prepare_data(device)

        _list_relative = getattr(
            self._load_module(device.type, configs_i.size(1), self.particle_cut),
            "list_relative",
        )
        configs_j, psi_j_real = _list_relative(
            configs_i, torch.view_as_real(psi_i), self.site, self.kind, self.coef, configs_exclude
        )
        return configs_j, torch.view_as_complex(psi_j_real)

    def diagonal_term(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Get the diagonal term of the Hamiltonian for the given configurations.

        Parameters
        ----------
        configs : torch.Tensor
            A uint8 tensor of shape [batch_size, n_qubytes] representing the input configurations.

        Returns
        -------
        torch.Tensor
            A complex64 tensor of shape [batch_size] representing the diagonal term of the Hamiltonian for the given configurations.
        """
        world_size = get_world_size()
        rank = get_rank()
        device = get_local_device()

        if world_size == 1:
            return self._diagonal_term_local(configs, device)

        # Distributed computation
        batch_size = configs.size(0)
        chunk_size = batch_size // world_size
        remainder = batch_size % world_size

        start_idx = rank * chunk_size + min(rank, remainder)
        end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)

        if start_idx >= end_idx:
            local_psi = torch.empty(0, dtype=torch.complex64, device=device)
        else:
            configs_chunk = configs[start_idx:end_idx]
            local_psi = self._diagonal_term_local(configs_chunk, device)

        if rank == 0:
            # Send RPC requests to all other ranks FIRST (asynchronous)
            rpc_refs = []
            for target_rank in range(1, world_size):
                rpc_ref = rpc.remote(
                    f"rank_{target_rank}",
                    _remote_diagonal_term,
                    args=(
                        self._get_data_tuple(),
                        configs.to("cpu"),
                        target_rank,
                        batch_size,
                        world_size,
                    ),
                )
                rpc_refs.append(rpc_ref)

            # Now compute local chunk while RPC is running remotely
            if start_idx >= end_idx:
                local_psi = torch.empty(0, dtype=torch.complex64, device=device)
            else:
                configs_chunk = configs[start_idx:end_idx]
                local_psi = self._diagonal_term_local(configs_chunk, device)

            # Collect results from all RPC calls
            results = [(local_psi, start_idx)]
            for rpc_ref in rpc_refs:
                remote_psi, remote_start_idx = rpc_ref.to_here()
                results.append((remote_psi.to(device), remote_start_idx))

            # Sort by start_idx and concatenate
            results.sort(key=lambda x: x[1])
            final_psi = torch.cat([r[0] for r in results if r[0].size(0) > 0], dim=0)
            return final_psi
        else:
            return local_psi

    def _diagonal_term_local(
        self,
        configs: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Local diagonal_term computation on a specific device.
        """
        configs = configs.to(device=device)
        self._prepare_data(device)

        _diagonal_term = getattr(
            self._load_module(device.type, configs.size(1), self.particle_cut),
            "diagonal_term",
        )
        psi_result = torch.view_as_complex(_diagonal_term(configs, self.site, self.kind, self.coef))
        return psi_result


# ============================================================================
# Remote computation functions (called via RPC on workers)
# ============================================================================

def _create_hamiltonian_from_data(data_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]) -> Hamiltonian:
    """
    Create a Hamiltonian instance from serialized data tuple.
    """
    site, kind, coef, particle_cut = data_tuple
    hamiltonian = Hamiltonian((site, kind, coef), kind="fermi" if particle_cut == 1 else "bose2")
    return hamiltonian


def _remote_apply_within(
    data_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    configs_i: torch.Tensor,
    psi_i: torch.Tensor,
    configs_j: torch.Tensor,
    target_rank: int,
    batch_size_i: int,
    world_size: int,
) -> torch.Tensor:
    """
    Remote apply_within computation on a worker.
    """
    rank = get_rank()

    device = get_local_device()
    hamiltonian = _create_hamiltonian_from_data(data_tuple)

    chunk_size = batch_size_i // world_size
    remainder = batch_size_i % world_size
    start_idx = target_rank * chunk_size + min(target_rank, remainder)
    end_idx = start_idx + chunk_size + (1 if target_rank < remainder else 0)

    if start_idx >= end_idx:
        return torch.zeros(configs_j.size(0), dtype=torch.complex64, device="cpu")

    configs_i_chunk = configs_i[start_idx:end_idx].to(device)
    psi_i_chunk = psi_i[start_idx:end_idx].to(device)
    configs_j_dev = configs_j.to(device)

    result = hamiltonian._apply_within_local(configs_i_chunk, psi_i_chunk, configs_j_dev, device)
    return result.to("cpu")


def _remote_find_relative(
    data_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    configs_i: torch.Tensor,
    psi_i: torch.Tensor,
    configs_exclude: torch.Tensor,
    target_rank: int,
    batch_size_i: int,
    world_size: int,
    count_selected: int,
) -> torch.Tensor:
    """
    Remote find_relative computation on a worker.
    """
    device = get_local_device()
    hamiltonian = _create_hamiltonian_from_data(data_tuple)

    chunk_size = batch_size_i // world_size
    remainder = batch_size_i % world_size
    start_idx = target_rank * chunk_size + min(target_rank, remainder)
    end_idx = start_idx + chunk_size + (1 if target_rank < remainder else 0)

    if start_idx >= end_idx:
        return torch.empty(0, configs_i.size(1), dtype=torch.uint8, device="cpu")

    configs_i_chunk = configs_i[start_idx:end_idx].to(device)
    psi_i_chunk = psi_i[start_idx:end_idx].to(device)
    configs_exclude_dev = configs_exclude.to(device)

    result = hamiltonian._find_relative_local(configs_i_chunk, psi_i_chunk, count_selected, configs_exclude_dev, device)
    return result.to("cpu")


def _remote_list_relative(
    data_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    configs_i: torch.Tensor,
    psi_i: torch.Tensor,
    configs_exclude: torch.Tensor,
    target_rank: int,
    batch_size_i: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remote list_relative computation on a worker.
    """
    device = get_local_device()
    hamiltonian = _create_hamiltonian_from_data(data_tuple)

    chunk_size = batch_size_i // world_size
    remainder = batch_size_i % world_size
    start_idx = target_rank * chunk_size + min(target_rank, remainder)
    end_idx = start_idx + chunk_size + (1 if target_rank < remainder else 0)

    if start_idx >= end_idx:
        return torch.empty(0, configs_i.size(1), dtype=torch.uint8, device="cpu"), torch.empty(0, dtype=torch.complex64, device="cpu")

    configs_i_chunk = configs_i[start_idx:end_idx].to(device)
    psi_i_chunk = psi_i[start_idx:end_idx].to(device)
    configs_exclude_dev = configs_exclude.to(device)

    configs_result, psi_result = hamiltonian._list_relative_local(configs_i_chunk, psi_i_chunk, configs_exclude_dev, device)
    return configs_result.to("cpu"), psi_result.to("cpu")


def _remote_diagonal_term(
    data_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    configs: torch.Tensor,
    target_rank: int,
    batch_size: int,
    world_size: int,
) -> tuple[torch.Tensor, int]:
    """
    Remote diagonal_term computation on a worker.
    """
    device = get_local_device()
    hamiltonian = _create_hamiltonian_from_data(data_tuple)

    chunk_size = batch_size // world_size
    remainder = batch_size % world_size
    start_idx = target_rank * chunk_size + min(target_rank, remainder)
    end_idx = start_idx + chunk_size + (1 if target_rank < remainder else 0)

    if start_idx >= end_idx:
        return torch.empty(0, dtype=torch.complex64, device="cpu"), start_idx

    configs_chunk = configs[start_idx:end_idx].to(device)
    result = hamiltonian._diagonal_term_local(configs_chunk, device)
    return result.to("cpu"), start_idx