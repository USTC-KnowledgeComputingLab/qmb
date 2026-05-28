import os
import warnings
from typing import Any, ClassVar, Literal

import platformdirs
import torch
import torch.utils.cpp_extension


class Hamiltonian:
    _compiled_modules: ClassVar[dict[tuple[str, int, int, int], object]] = {}

    def __init__(
        self,
        hamiltonian: dict,
        *,
        kind: Literal["fermi", "bose2"],
        max_op_number: int,
        devices: list[str],
    ) -> None:
        self._device = self._parse_device(devices)
        self._max_op_number = max_op_number
        match kind:
            case "fermi":
                self._particle_cut = 1
            case "bose2":
                self._particle_cut = 2
        self._site, self._kind, self._coef = self._prepare(hamiltonian, self._max_op_number)
        self._sort_site_kind_coef()
        self._site = self._site.to(device=self._device).contiguous()
        self._kind = self._kind.to(device=self._device).contiguous()
        self._coef = self._coef.to(device=self._device).contiguous()

    @staticmethod
    def _parse_device(devices: list[str]) -> torch.device:
        if len(devices) > 1:
            raise NotImplementedError("Multiple devices are not yet supported.")
        device_str = devices[0]
        parts = device_str.split(":")
        if len(parts) < 2 or len(parts) > 3 or parts[0] != "localhost":
            raise ValueError(f"Invalid device string: {device_str}")
        if parts[1] == "cpu":
            return torch.device("cpu")
        if parts[1] == "cuda" and len(parts) == 3:
            return torch.device(f"cuda:{parts[2]}")
        raise ValueError(f"Invalid device string: {device_str}")

    @classmethod
    def _set_torch_cuda_arch_list(cls) -> None:
        if not torch.cuda.is_available():
            return
        if "TORCH_CUDA_ARCH_LIST" in os.environ:
            return
        os.environ["TORCH_CUDA_ARCH_LIST"] = "native"

    @classmethod
    def _load_module(
        cls,
        device_type: str = "declaration",
        n_qubytes: int = 0,
        particle_cut: int = 0,
        max_op_number: int = 0,
    ) -> Any:
        cls._set_torch_cuda_arch_list()
        is_declaration = device_type == "declaration"
        if not is_declaration:
            cls._load_module("declaration", 0, 0, 0)
        key = (device_type, n_qubytes, particle_cut, max_op_number)
        name = (
            "qmp_hamiltonian"
            if is_declaration
            else f"qmp_hamiltonian_{n_qubytes}_{particle_cut}_{max_op_number}"
        )
        if key not in cls._compiled_modules:
            build_directory = platformdirs.user_cache_path("qmp", "kclab") / name / device_type
            build_directory.mkdir(parents=True, exist_ok=True)
            folder = os.path.dirname(__file__)
            match device_type:
                case "declaration":
                    sources = [os.path.join(folder, "_hamiltonian.cpp")]
                case "cpu":
                    sources = [os.path.join(folder, "_hamiltonian_cpu.cpp")]
                case "cuda":
                    sources = [os.path.join(folder, "_hamiltonian_cuda.cu")]
                case _:
                    raise ValueError(f"Unsupported device type: {device_type}")
            cls._compiled_modules[key] = torch.utils.cpp_extension.load(
                name=name,
                sources=sources,
                is_python_module=is_declaration,
                extra_cflags=[
                    "-O3",
                    "-ffast-math",
                    "-march=native",
                    f"-DN_QUBYTES={n_qubytes}",
                    f"-DPARTICLE_CUT={particle_cut}",
                    f"-DMAX_OP_NUMBER={max_op_number}",
                    "-std=c++20",
                ],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    f"-DN_QUBYTES={n_qubytes}",
                    f"-DPARTICLE_CUT={particle_cut}",
                    f"-DMAX_OP_NUMBER={max_op_number}",
                    "-std=c++20",
                ],
                build_directory=build_directory,
            )
        if is_declaration:
            return cls._compiled_modules[key]
        return getattr(torch.ops, name)

    @classmethod
    def _prepare(cls, hamiltonian: dict, max_op_number: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return cls._load_module().prepare(hamiltonian, max_op_number)

    def _sort_site_kind_coef(self) -> None:
        order = self._coef.norm(dim=1).argsort(descending=True)
        self._site = self._site[order]
        self._kind = self._kind[order]
        self._coef = self._coef[order]

    def apply_within_subspace_in_double_side(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        configs_j: torch.Tensor,
        *,
        configs_i_sorted: bool = False,
        configs_j_sorted: bool = False,
        direction: Literal["forward", "backward"] = "forward",
    ) -> torch.Tensor:
        configs_i = configs_i.to(device=self._device)
        psi_i = psi_i.to(device=self._device)
        configs_j = configs_j.to(device=self._device)
        op_module = self._load_module(
            self._device.type, configs_i.size(1), self._particle_cut, self._max_op_number
        )
        op = op_module.apply_within_subspace_in_double_side
        direction_flag = 0 if direction == "forward" else 1
        raw = op(
            configs_i,
            torch.view_as_real(psi_i).to(torch.float64),
            configs_j,
            self._site,
            self._kind,
            self._coef,
            configs_i_sorted,
            configs_j_sorted,
            direction_flag,
        )
        return torch.view_as_complex(raw).to(psi_i.dtype)

    @warnings.deprecated("use apply_within_subspace_in_double_side")
    def apply_within(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self.apply_within_subspace_in_double_side(*args, **kwargs)
