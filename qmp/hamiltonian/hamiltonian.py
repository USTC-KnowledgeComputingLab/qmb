"""
This file contains the Hamiltonian class, which is used to store the Hamiltonian and process iteration over each term in the Hamiltonian for given configurations.
"""

import os
import typing
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
        self._prepare_data(configs_i.device)
        _apply_within: typing.Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        _apply_within = getattr(
            self._load_module(configs_i.device.type, configs_i.size(1), self.particle_cut),
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
        self._prepare_data(configs_i.device)
        _find_relative: typing.Callable[
            [torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        _find_relative = getattr(
            self._load_module(configs_i.device.type, configs_i.size(1), self.particle_cut),
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
        self._prepare_data(configs_i.device)
        _list_relative = getattr(
            self._load_module(configs_i.device.type, configs_i.size(1), self.particle_cut),
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
        self._prepare_data(configs.device)
        _diagonal_term: typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        _diagonal_term = getattr(
            self._load_module(configs.device.type, configs.size(1), self.particle_cut),
            "diagonal_term",
        )
        psi_result = torch.view_as_complex(_diagonal_term(configs, self.site, self.kind, self.coef))
        return psi_result
