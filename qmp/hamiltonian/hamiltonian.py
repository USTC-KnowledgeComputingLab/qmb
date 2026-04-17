"""
This file contains the Hamiltonian class, which is used to store the Hamiltonian and process iteration over each term in the Hamiltonian for given configurations.
"""

import os
import logging
import platformdirs
import torch
import torch.utils.cpp_extension


class Hamiltonian:
    """
    The Hamiltonian type, which stores the Hamiltonian and processes iteration over each term in the Hamiltonian for given configurations.
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

        # Multi-device support: cache tensors on each device
        # These dictionaries store site, kind, coef tensors for each device
        self._site_dict: dict[str, torch.Tensor] = {}
        self._kind_dict: dict[str, torch.Tensor] = {}
        self._coef_dict: dict[str, torch.Tensor] = {}

    def _sort_site_kind_coef(self) -> None:
        """
        Reorder the site, kind, and coefficient tensors in descending order of the norm of the coefficients.
        """
        order = self.coef.norm(dim=1).argsort(descending=True)
        self.site = self.site[order]
        self.kind = self.kind[order]
        self.coef = self.coef[order]

    def _prepare_data_for_device(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare the site, kind, and coefficient tensors for computation on a specific device.
        Uses caching to avoid repeated transfers for the same device.
        """
        device_key = str(device)
        if device_key not in self._site_dict:
            self._site_dict[device_key] = self.site.to(device=device).contiguous()
            self._kind_dict[device_key] = self.kind.to(device=device).contiguous()
            self._coef_dict[device_key] = self.coef.to(device=device).contiguous()
        return self._site_dict[device_key], self._kind_dict[device_key], self._coef_dict[device_key]

    def _prepare_data(self, device: torch.device) -> None:
        """
        Prepare the site, kind, and coefficient tensors for computation on the given device.
        Deprecated: use _prepare_data_for_device instead for multi-device support.
        """
        self.site = self.site.to(device=device).contiguous()
        self.kind = self.kind.to(device=device).contiguous()
        self.coef = self.coef.to(device=device).contiguous()

    def apply_within(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        configs_j: torch.Tensor,
        devices: list[torch.device] | None = None,
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
        devices : list[torch.device] | None
            A list of devices to use for computation. If None, uses the device of configs_i.

        Returns
        -------
        torch.Tensor
            A tensor of shape [batch_size_j] representing the output amplitudes on the given configurations.
        """
        if devices is None or len(devices) == 1:
            # Single device case: use original implementation
            device = devices[0] if devices else configs_i.device
            site, kind, coef = self._prepare_data_for_device(device)
            configs_i_dev = configs_i.to(device=device)
            psi_i_dev = psi_i.to(device=device)
            configs_j_dev = configs_j.to(device=device)
            _apply_within = getattr(
                self._load_module(device.type, configs_i_dev.size(1), self.particle_cut),
                "apply_within",
            )
            psi_j = torch.view_as_complex(
                _apply_within(configs_i_dev, torch.view_as_real(psi_i_dev), configs_j_dev, site, kind, coef)
            )
            return psi_j

        # Multi-device case: split configs_i across devices and accumulate results
        n_devices = len(devices)
        batch_size_i = configs_i.size(0)
        chunk_size = batch_size_i // n_devices
        remainder = batch_size_i % n_devices

        results: list[torch.Tensor] = []
        for idx, device in enumerate(devices):
            # Calculate chunk indices
            start_idx = idx * chunk_size + min(idx, remainder)
            end_idx = start_idx + chunk_size + (1 if idx < remainder else 0)

            # Skip empty chunks
            if start_idx >= end_idx:
                continue

            # Prepare data for this device
            site, kind, coef = self._prepare_data_for_device(device)

            # Move chunk to device
            configs_i_chunk = configs_i[start_idx:end_idx].to(device=device)
            psi_i_chunk = psi_i[start_idx:end_idx].to(device=device)
            configs_j_dev = configs_j.to(device=device)

            # Execute on device
            _apply_within = getattr(
                self._load_module(device.type, configs_i_chunk.size(1), self.particle_cut),
                "apply_within",
            )
            psi_j_chunk = torch.view_as_complex(
                _apply_within(configs_i_chunk, torch.view_as_real(psi_i_chunk), configs_j_dev, site, kind, coef)
            )
            results.append(psi_j_chunk)

        # Accumulate results from all devices
        # All results are tensors of shape [batch_size_j], we sum them
        result_device = devices[0]
        final_result = torch.zeros(configs_j.size(0), dtype=torch.complex64, device=result_device)
        for r in results:
            final_result = final_result + r.to(device=result_device)
        return final_result

    def find_relative(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        count_selected: int,
        configs_exclude: torch.Tensor | None = None,
        devices: list[torch.device] | None = None,
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
        devices : list[torch.device] | None
            A list of devices to use for computation. If None, uses the device of configs_i.

        Returns
        -------
        torch.Tensor
            The resulting configurations after applying the Hamiltonian, only the first `count_selected` configurations are guaranteed to be returned.
            The order of the configurations is guaranteed to be sorted by estimated psi for the remaining configurations.
        """
        if configs_exclude is None:
            configs_exclude = configs_i

        if devices is None or len(devices) == 1:
            # Single device case: use original implementation
            device = devices[0] if devices else configs_i.device
            site, kind, coef = self._prepare_data_for_device(device)
            configs_i_dev = configs_i.to(device=device)
            psi_i_dev = psi_i.to(device=device)
            configs_exclude_dev = configs_exclude.to(device=device)
            _find_relative = getattr(
                self._load_module(device.type, configs_i_dev.size(1), self.particle_cut),
                "find_relative",
            )
            configs_j = _find_relative(
                configs_i_dev, torch.view_as_real(psi_i_dev), count_selected, site, kind, coef, configs_exclude_dev
            )
            return configs_j

        # Multi-device case: split configs_i across devices and merge results
        n_devices = len(devices)
        batch_size_i = configs_i.size(0)
        chunk_size = batch_size_i // n_devices
        remainder = batch_size_i % n_devices

        results: list[torch.Tensor] = []
        for idx, device in enumerate(devices):
            # Calculate chunk indices
            start_idx = idx * chunk_size + min(idx, remainder)
            end_idx = start_idx + chunk_size + (1 if idx < remainder else 0)

            # Skip empty chunks
            if start_idx >= end_idx:
                continue

            # Prepare data for this device
            site, kind, coef = self._prepare_data_for_device(device)

            # Move chunk to device
            configs_i_chunk = configs_i[start_idx:end_idx].to(device=device)
            psi_i_chunk = psi_i[start_idx:end_idx].to(device=device)
            configs_exclude_dev = configs_exclude.to(device=device)

            # Execute on device
            _find_relative = getattr(
                self._load_module(device.type, configs_i_chunk.size(1), self.particle_cut),
                "find_relative",
            )
            # Request count_selected per device, we'll merge and select top later
            configs_j_chunk = _find_relative(
                configs_i_chunk, torch.view_as_real(psi_i_chunk), count_selected, site, kind, coef, configs_exclude_dev
            )
            results.append(configs_j_chunk.to(device=devices[0]))

        # Merge results while preserving order (importance ranking from each GPU)
        # torch.unique with sorted=True returns unique in order of first occurrence
        # This preserves the relative importance ordering from each GPU
        all_configs = torch.cat(results, dim=0)
        unique_configs = torch.unique(all_configs, sorted=True, dim=0)

        # Exclude configs that appear in configs_exclude
        # Build a mask to filter out excluded configurations
        result_device = devices[0]
        configs_exclude_dev = configs_exclude.to(device=result_device)

        # Check if each unique_config is in configs_exclude
        # Use broadcasting to compare: unique_configs vs configs_exclude
        # Shape: [n_unique, 1, n_qubytes] vs [1, n_exclude, n_qubytes]
        # Result: [n_unique, n_exclude] boolean matrix, then check if any match
        exclude_mask = (unique_configs.unsqueeze(1) == configs_exclude_dev.unsqueeze(0)).all(dim=-1).any(dim=-1)
        # Keep configurations that are NOT in configs_exclude
        filtered_configs = unique_configs[~exclude_mask]

        # Return up to count_selected
        if filtered_configs.size(0) > count_selected:
            filtered_configs = filtered_configs[:count_selected]
        return filtered_configs

    def list_relative(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        configs_exclude: torch.Tensor | None = None,
        devices: list[torch.device] | None = None,
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
        devices : list[torch.device] | None
            A list of devices to use for computation. If None, uses the device of configs_i.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (configs_j, psi_j) where configs_j are unique new configurations
            and psi_j are their summed amplitudes from all connected paths.
        """
        if configs_exclude is None:
            configs_exclude = configs_i

        if devices is None or len(devices) == 1:
            # Single device case: use original implementation
            device = devices[0] if devices else configs_i.device
            site, kind, coef = self._prepare_data_for_device(device)
            configs_i_dev = configs_i.to(device=device)
            psi_i_dev = psi_i.to(device=device)
            configs_exclude_dev = configs_exclude.to(device=device)
            _list_relative = getattr(
                self._load_module(device.type, configs_i_dev.size(1), self.particle_cut),
                "list_relative",
            )
            configs_j, psi_j_real = _list_relative(
                configs_i_dev, torch.view_as_real(psi_i_dev), site, kind, coef, configs_exclude_dev
            )
            return configs_j, torch.view_as_complex(psi_j_real)

        # Multi-device case: split configs_i across devices and merge results
        n_devices = len(devices)
        batch_size_i = configs_i.size(0)
        chunk_size = batch_size_i // n_devices
        remainder = batch_size_i % n_devices

        results_configs: list[torch.Tensor] = []
        results_psi: list[torch.Tensor] = []
        for idx, device in enumerate(devices):
            # Calculate chunk indices
            start_idx = idx * chunk_size + min(idx, remainder)
            end_idx = start_idx + chunk_size + (1 if idx < remainder else 0)

            # Skip empty chunks
            if start_idx >= end_idx:
                continue

            # Prepare data for this device
            site, kind, coef = self._prepare_data_for_device(device)

            # Move chunk to device
            configs_i_chunk = configs_i[start_idx:end_idx].to(device=device)
            psi_i_chunk = psi_i[start_idx:end_idx].to(device=device)
            configs_exclude_dev = configs_exclude.to(device=device)

            # Execute on device
            _list_relative = getattr(
                self._load_module(device.type, configs_i_chunk.size(1), self.particle_cut),
                "list_relative",
            )
            configs_j_chunk, psi_j_chunk_real = _list_relative(
                configs_i_chunk, torch.view_as_real(psi_i_chunk), site, kind, coef, configs_exclude_dev
            )
            psi_j_chunk = torch.view_as_complex(psi_j_chunk_real)
            results_configs.append(configs_j_chunk.to(device=devices[0]))
            results_psi.append(psi_j_chunk.to(device=devices[0]))

        # Merge results: concatenate and deduplicate with amplitude accumulation
        all_configs = torch.cat(results_configs, dim=0)
        all_psi = torch.cat(results_psi, dim=0)

        # Deduplicate and sum amplitudes
        result_device = devices[0]
        unique_configs, inverse_indices = torch.unique(all_configs, return_inverse=True, dim=0)
        # Sum amplitudes for each unique configuration
        unique_psi = torch.zeros(unique_configs.size(0), dtype=all_psi.dtype, device=result_device)
        unique_psi.scatter_add_(0, inverse_indices, all_psi)

        # Exclude configs that appear in configs_exclude
        configs_exclude_dev = configs_exclude.to(device=result_device)
        # Check if each unique_config is in configs_exclude
        exclude_mask = (unique_configs.unsqueeze(1) == configs_exclude_dev.unsqueeze(0)).all(dim=-1).any(dim=-1)
        # Keep configurations that are NOT in configs_exclude
        filtered_configs = unique_configs[~exclude_mask]
        filtered_psi = unique_psi[~exclude_mask]

        return filtered_configs, filtered_psi

    def diagonal_term(self, configs: torch.Tensor, devices: list[torch.device] | None = None) -> torch.Tensor:
        """
        Get the diagonal term of the Hamiltonian for the given configurations.

        Parameters
        ----------
        configs : torch.Tensor
            A uint8 tensor of shape [batch_size, n_qubytes] representing the input configurations.
        devices : list[torch.device] | None
            A list of devices to use for computation. If None, uses the device of configs.

        Returns
        -------
        torch.Tensor
            A complex64 tensor of shape [batch_size] representing the diagonal term of the Hamiltonian for the given configurations.
        """
        if devices is None or len(devices) == 1:
            # Single device case: use original implementation
            device = devices[0] if devices else configs.device
            site, kind, coef = self._prepare_data_for_device(device)
            configs_dev = configs.to(device=device)
            _diagonal_term = getattr(
                self._load_module(device.type, configs_dev.size(1), self.particle_cut),
                "diagonal_term",
            )
            psi_result = torch.view_as_complex(_diagonal_term(configs_dev, site, kind, coef))
            return psi_result

        # Multi-device case: split configs across devices and concatenate results
        n_devices = len(devices)
        batch_size = configs.size(0)
        chunk_size = batch_size // n_devices
        remainder = batch_size % n_devices

        results: list[torch.Tensor] = []
        for idx, device in enumerate(devices):
            # Calculate chunk indices
            start_idx = idx * chunk_size + min(idx, remainder)
            end_idx = start_idx + chunk_size + (1 if idx < remainder else 0)

            # Skip empty chunks
            if start_idx >= end_idx:
                continue

            # Prepare data for this device
            site, kind, coef = self._prepare_data_for_device(device)

            # Move chunk to device
            configs_chunk = configs[start_idx:end_idx].to(device=device)

            # Execute on device
            _diagonal_term = getattr(
                self._load_module(device.type, configs_chunk.size(1), self.particle_cut),
                "diagonal_term",
            )
            psi_chunk = torch.view_as_complex(_diagonal_term(configs_chunk, site, kind, coef))
            results.append(psi_chunk)

        # Concatenate results
        return torch.cat(results, dim=0).to(device=devices[0])
