# apply_within_subspace_in_double_side Implementation Plan

> **Status**: completed. Implemented with modifications — device ownership inverted, C++ kernel/interface restructured, parameter ordering unified. See spec for final design.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Hamiltonian `apply_within_subspace_in_double_side` — H|ψ⟩ projected onto configs_j, with CPU+CUDA backends.

**Architecture:** Python `Hamiltonian` wraps JIT-compiled C++/CUDA. `MAX_OP_NUMBER`, `N_QUBYTES`, `PARTICLE_CUT` as compile-time macros. JW sign via 256-byte constexpr parity table. Forward/Backward as template parameter.

**Tech Stack:** Python 3.13+, hatchling+hatch-vcs, PyTorch, pybind11, C++20

---

### Task 0: Bump Python requirement to 3.13

**Files:** Modify `pyproject.toml`

- [ ] **Step 1:** Change `requires-python = ">=3.12"` to `requires-python = ">=3.13"` in pyproject.toml
- [ ] **Step 2:** Commit `chore: bump requires-python to >=3.13`

---

### Task 1: Create Hamiltonian package skeleton

**Files:** Create `src/qmp/hamiltonian/__init__.py`

```python
from qmp.hamiltonian._hamiltonian import Hamiltonian
__all__ = ["Hamiltonian"]
```

- [ ] **Step 1:** `mkdir -p src/qmp/hamiltonian && mkdir -p tests`
- [ ] **Step 2:** Write `__init__.py` and empty `tests/__init__.py`
- [ ] **Step 3:** Commit `feat: add hamiltonian package skeleton`

---

### Task 2: Write tests (RED)

**Files:** Create `tests/test_hamiltonian.py`

```python
import pytest
import torch
import warnings
from qmp.hamiltonian import Hamiltonian


def _build_two_site_fermion():
    """H = c†₁c₀ + c†₀c₁, 2-site spinless fermion."""
    return Hamiltonian(
        {((1, 1), (0, 0)): -1.0, ((0, 1), (1, 0)): -1.0},
        kind="fermi",
        devices=["localhost:cpu:0"],
    )


def _build_two_site_boson():
    """Same hopping, bose2 (hard-core boson, no JW sign)."""
    return Hamiltonian(
        {((1, 1), (0, 0)): -1.0, ((0, 1), (1, 0)): -1.0},
        kind="bose2",
        devices=["localhost:cpu:0"],
    )


def _build_hubbard_2x2():
    """
    4-site spinful Hubbard: 8 spin-orbitals in 1 byte.
    Orbitals: 0=1↑, 1=1↓, 2=2↑, 3=2↓, 4=3↑, 5=3↓, 6=4↑, 7=4↓
    2×2 lattice: sites 1-2 horizontal, 1-3 vertical, 2-4 vertical, 3-4 horizontal
    H = -t Σ_{⟨i,j⟩,σ} c†_{jσ}c_{iσ} + h.c.  +  U Σ_i n_{i↑}n_{i↓}
       = -t Σ (c†_{2↑}c_{1↑} + c†_{1↑}c_{2↑} + c†_{2↓}c_{1↓} + c†_{1↓}c_{2↓}
             + c†_{3↑}c_{1↑} + c†_{1↑}c_{3↑} + c†_{3↓}c_{1↓} + c†_{1↓}c_{3↓}
             + c†_{4↑}c_{2↑} + c†_{2↑}c_{4↑} + c†_{4↓}c_{2↓} + c†_{2↓}c_{4↓}
             + c†_{4↑}c_{3↑} + c†_{3↑}c_{4↑} + c†_{4↓}c_{3↓} + c†_{3↓}c_{4↓})
       + U (n_{1↑}n_{1↓} + n_{2↑}n_{2↓} + n_{3↑}n_{3↓} + n_{4↑}n_{4↓})

    Each n_{i↑}n_{i↓} = c†_{i↑}c_{i↑}c†_{i↓}c_{i↓}
    """
    t_val = 1.0
    u_val = 4.0
    ham = {}

    def idx(site, spin):
        return site * 2 + spin  # spin=0 ↑, spin=1 ↓

    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]  # 0-indexed sites
    for i, j in edges:
        for s in (0, 1):
            si = idx(i, s)
            sj = idx(j, s)
            ham[((sj, 1), (si, 0))] = -t_val
            ham[((si, 1), (sj, 0))] = -t_val  # h.c.

    for site in range(4):
        up = idx(site, 0)
        dn = idx(site, 1)
        ham[((up, 1), (up, 0), (dn, 1), (dn, 0))] = u_val

    return Hamiltonian(ham, kind="fermi", devices=["localhost:cpu:0"])


class TestBasicFermion:
    def test_single_hopping_forward(self):
        """H|c†₀|vac⟩ = -|c†₁|vac⟩"""
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert pj.shape == (1,)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))

    def test_hermitian_conjugate(self):
        """H|10⟩ = -|01⟩"""
        h = _build_two_site_fermion()
        ci = torch.tensor([[2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))

    def test_pauli_exclusion(self):
        """|11⟩: both occupied, creating fails → zero."""
        h = _build_two_site_fermion()
        ci = torch.tensor([[3]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[3]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([0.0 + 0.0j], dtype=torch.complex64))

    def test_no_connected_term(self):
        """|1⟩ not connected to |11⟩ by single hopping."""
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[3]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([0.0 + 0.0j], dtype=torch.complex64))

    def test_multiple_contributions(self):
        """Two terms hit same target, amplitudes accumulate."""
        h = Hamiltonian(
            {((1, 1), (0, 0)): -2.0, ((0, 1), (1, 0)): -1.0},
            kind="fermi",
            devices=["localhost:cpu:0"],
        )
        ci = torch.tensor([[1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1], [2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        # |1⟩ from term1 × |2⟩: (-1.0)*(0.5)= -0.5
        # |2⟩ from term0 × |1⟩: (-2.0)*(1.0)= -2.0
        assert torch.allclose(pj, torch.tensor([-0.5 + 0.0j, -2.0 + 0.0j], dtype=torch.complex64))

    def test_complex_coefficients(self):
        """Complex coefficient produces complex output."""
        h = Hamiltonian(
            {((1, 1), (0, 0)): 0.0 + 1.0j},
            kind="fermi",
            devices=["localhost:cpu:0"],
        )
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([0.0 + 1.0j], dtype=torch.complex64))


class TestBoson:
    def test_no_jw_sign(self):
        h = _build_two_site_boson()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))


class TestDirection:
    def test_forward_matches_backward(self):
        h = _build_two_site_fermion()
        ci = torch.tensor([[1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1], [2]], dtype=torch.uint8)
        fwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="forward")
        bwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="backward")
        assert torch.allclose(fwd, bwd)

    def test_forward_matches_backward_hubbard(self):
        h = _build_hubbard_2x2()
        ci = torch.tensor([[0b00000011]], dtype=torch.uint8)  # |1↑1↓⟩
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00001100]], dtype=torch.uint8)  # |2↑2↓⟩
        fwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="forward")
        bwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="backward")
        assert torch.allclose(fwd, bwd)


class TestDeprecation:
    def test_apply_within_warns(self):
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        with pytest.warns(DeprecationWarning, match="apply_within_subspace_in_double_side"):
            result = h.apply_within(ci, pi, cj)
        expected = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(result, expected)


class TestDevices:
    def test_multi_device_raises(self):
        h = Hamiltonian(
            {((1, 1), (0, 0)): -1.0},
            kind="fermi",
            devices=["localhost:cuda:0", "localhost:cuda:1"],
        )
        ci = torch.tensor([[1]], dtype=torch.uint8)
        with pytest.raises(NotImplementedError):
            h.apply_within_subspace_in_double_side(
                ci, torch.tensor([1.0 + 0.0j], dtype=torch.complex64), ci
            )


class TestSorted:
    def test_sorted_params_no_crash(self):
        """Both sorted flags set, should not crash (no-op for sort)."""
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(
            ci, pi, cj, configs_i_sorted=True, configs_j_sorted=True
        )
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))


class TestHubbard:
    def test_diagonal_u_term(self):
        """U term: n_{i↑}n_{i↓} on a doubly occupied site gives +U."""
        h = _build_hubbard_2x2()
        ci = torch.tensor([[0b00000011]], dtype=torch.uint8)  # |1↑1↓⟩: bits 0,1
        cj = torch.tensor([[0b00000011]], dtype=torch.uint8)  # same config
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        # U=4, 4 operators: c†↑ c↑ c†↓ c↓ → creates then destroys = identity, sign +1
        assert torch.allclose(
            torch.abs(pj), torch.tensor([4.0], dtype=torch.float32), atol=1e-5
        )

    def test_hopping_preserves_spin(self):
        """Hopping t=1: H|c†₁↑|vac⟩ has -|c†₂↑|vac⟩ + -|c†₃↑|vac⟩."""
        h = _build_hubbard_2x2()
        ci = torch.tensor([[0b00000001]], dtype=torch.uint8)  # |1↑⟩
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor(
            [[0b00000100], [0b00010000]], dtype=torch.uint8
        )  # |2↑⟩, |3↑⟩
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j, -1.0 + 0.0j], dtype=torch.complex64))
```

- [ ] **Step 1:** Write test file
- [ ] **Step 2:** Run `PYTHONPATH=src python -m pytest tests/test_hamiltonian.py -v` → expect ImportError
- [ ] **Step 3:** Commit `test: add hamiltonian tests (RED)`

---

### Task 3: Implement `_hamiltonian.py` — Python class

**Files:** Create `src/qmp/hamiltonian/_hamiltonian.py`

```python
import os
import warnings
from typing import Literal

import platformdirs
import torch
import torch.utils.cpp_extension


class Hamiltonian:
    _compiled_modules: dict[tuple[str, int, int, int], object] = {}

    def __init__(
        self,
        hamiltonian: dict,
        *,
        kind: Literal["fermi", "bose2"],
        devices: list[str],
        max_op_number: int = 4,
    ) -> None:
        self._devices = devices
        self._max_op_number = max_op_number
        match kind:
            case "fermi":
                self._particle_cut: int = 1
            case "bose2":
                self._particle_cut: int = 2
        self._site, self._kind, self._coef = self._prepare(hamiltonian)
        self._sort_site_kind_coef()
        self._device_data: dict[torch.device, None] = {}

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
        max_op_number: int = 0,
        n_qubytes: int = 0,
        particle_cut: int = 0,
    ) -> object:
        cls._set_torch_cuda_arch_list()
        is_declaration = device_type == "declaration"
        if not is_declaration:
            cls._load_module("declaration", 0, 0, 0)
        key = (device_type, max_op_number, n_qubytes, particle_cut)
        name = (
            "qmp_hamiltonian"
            if is_declaration
            else f"qmp_hamiltonian_{max_op_number}_{n_qubytes}_{particle_cut}"
        )
        if key not in cls._compiled_modules:
            build_directory = (
                platformdirs.user_cache_path("qmp", "kclab") / name / device_type
            )
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
                    "-O3", "-ffast-math", "-march=native",
                    f"-DMAX_OP_NUMBER={max_op_number}",
                    f"-DN_QUBYTES={n_qubytes}",
                    f"-DPARTICLE_CUT={particle_cut}",
                    "-std=c++20",
                ],
                extra_cuda_cflags=[
                    "-O3", "--use_fast_math",
                    f"-DMAX_OP_NUMBER={max_op_number}",
                    f"-DN_QUBYTES={n_qubytes}",
                    f"-DPARTICLE_CUT={particle_cut}",
                    "-std=c++20",
                ],
                build_directory=build_directory,
            )
        if is_declaration:
            return cls._compiled_modules[key]
        return getattr(torch.ops, name)

    @classmethod
    def _prepare(
        cls, hamiltonian: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return getattr(cls._load_module(), "prepare")(hamiltonian)

    def _sort_site_kind_coef(self) -> None:
        order = self._coef.norm(dim=1).argsort(descending=True)
        self._site = self._site[order]
        self._kind = self._kind[order]
        self._coef = self._coef[order]

    def _move_to_device(self, device: torch.device) -> None:
        if device in self._device_data:
            return
        self._site = self._site.to(device=device).contiguous()
        self._kind = self._kind.to(device=device).contiguous()
        self._coef = self._coef.to(device=device).contiguous()
        self._device_data[device] = None

    def _ensure_single_device(self) -> torch.device:
        if len(self._devices) > 1:
            raise NotImplementedError("Multiple devices are not yet supported.")
        device_str = self._devices[0]
        parts = device_str.split(":")
        if parts[0] != "localhost":
            raise NotImplementedError(
                f"Non-localhost devices not yet supported: {device_str}"
            )
        if len(parts) == 2 and parts[1] == "cpu":
            return torch.device("cpu")
        if len(parts) >= 3:
            return torch.device(f"cuda:{parts[2]}")
        raise ValueError(f"Invalid device string: {device_str}")

    def _get_device_from_tensor(self, tensor: torch.Tensor) -> torch.device:
        if "cpu" in self._devices[0]:
            return torch.device("cpu")
        return tensor.device

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
        self._ensure_single_device()
        device = self._get_device_from_tensor(configs_i)
        self._move_to_device(device)
        op = getattr(
            self._load_module(
                device.type, self._max_op_number, configs_i.size(1), self._particle_cut
            ),
            "apply_within_subspace_in_double_side",
        )
        direction_flag = 0 if direction == "forward" else 1
        return torch.view_as_complex(
            op(
                configs_i, torch.view_as_real(psi_i), configs_j,
                self._site, self._kind, self._coef,
                configs_i_sorted, configs_j_sorted, direction_flag,
            )
        )

    @warnings.deprecated("use apply_within_subspace_in_double_side")
    def apply_within(self, *args, **kwargs):
        return self.apply_within_subspace_in_double_side(*args, **kwargs)
```

- [ ] **Step 1:** Write `_hamiltonian.py`
- [ ] **Step 2:** Commit `feat: add Hamiltonian Python wrapper`

---

### Task 4: Implement `_hamiltonian.cpp` — Declaration + prepare

**Files:** Create `src/qmp/hamiltonian/_hamiltonian.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <complex>
#include <cstdint>
#include <tuple>
#include <vector>

constexpr std::int64_t kMaxOpNumberFallback = 4;

template <std::int64_t max_op_number>
auto prepare(pybind11::dict hamiltonian)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
    std::vector<std::tuple<std::vector<std::int16_t>, std::vector<std::uint8_t>,
                           std::array<double, 2>>> terms;
    std::int64_t max_ops = 0;
    for (auto item : hamiltonian) {
        auto key_tuple = item.first.cast<std::tuple<pybind11::tuple>>();
        auto ops = std::get<0>(key_tuple);
        auto coef_val = item.second.cast<std::complex<double>>();
        std::vector<std::int16_t> sites;
        std::vector<std::uint8_t> kinds;
        for (auto op_item : ops) {
            auto op = op_item.cast<std::tuple<std::int64_t, std::int64_t>>();
            sites.push_back(static_cast<std::int16_t>(std::get<0>(op)));
            kinds.push_back(static_cast<std::uint8_t>(std::get<1>(op)));
        }
        max_ops = std::max(max_ops, static_cast<std::int64_t>(sites.size()));
        terms.emplace_back(std::move(sites), std::move(kinds),
                           std::array<double, 2>{{coef_val.real(), coef_val.imag()}});
    }
    auto eff_max = std::max(kMaxOpNumberFallback, max_ops);
    auto term_number = static_cast<std::int64_t>(terms.size());
    auto site = torch::zeros({term_number, eff_max},
                             torch::TensorOptions().dtype(torch::kInt16));
    auto kind = torch::full({term_number, eff_max}, 2,
                            torch::TensorOptions().dtype(torch::kUInt8));
    auto coef = torch::zeros({term_number, 2},
                             torch::TensorOptions().dtype(torch::kFloat64));
    for (std::int64_t i = 0; i < term_number; ++i) {
        const auto& [s, k, c] = terms[i];
        auto site_a = site[i];
        auto kind_a = kind[i];
        for (std::size_t j = 0; j < s.size(); ++j) {
            site_a[j] = s[j];
            kind_a[j] = k[j];
        }
        coef[i][0] = c[0];
        coef[i][1] = c[1];
    }
    return std::make_tuple(site, kind, coef);
}

#ifndef MAX_OP_NUMBER
#define MAX_OP_NUMBER 0
#endif
#ifndef N_QUBYTES
#define N_QUBYTES 0
#endif
#ifndef PARTICLE_CUT
#define PARTICLE_CUT 0
#endif

#define QMP_LIBRARY_HELPER(mo, nq, pc) qmp_hamiltonian_##mo##_##nq##_##pc
#define QMP_LIBRARY(mo, nq, pc) QMP_LIBRARY_HELPER(mo, nq, pc)

#if N_QUBYTES == 0
PYBIND11_MODULE(qmp_hamiltonian, m) {
    m.def("prepare", &prepare<kMaxOpNumberFallback>);
}
#else
TORCH_LIBRARY_FRAGMENT(QMP_LIBRARY(MAX_OP_NUMBER, N_QUBYTES, PARTICLE_CUT), m) {
    m.def("apply_within_subspace_in_double_side("
          "Tensor configs_i, Tensor psi_i, Tensor configs_j, "
          "Tensor site, Tensor kind, Tensor coef, "
          "bool configs_i_sorted, bool configs_j_sorted, int direction) -> Tensor");
}
#endif

#undef QMP_LIBRARY
#undef QMP_LIBRARY_HELPER
```

- [ ] **Step 1:** Write `_hamiltonian.cpp`
- [ ] **Step 2:** Commit `feat: add hamiltonian declaration module`

---

### Task 5: Implement `_hamiltonian_cpu.cpp` — CPU backend

**Files:** Create `src/qmp/hamiltonian/_hamiltonian_cpu.cpp`

```cpp
#include <torch/extension.h>
#include <array>
#include <algorithm>
#include <cstdint>
#include <utility>

namespace {

constexpr auto kParityTable = []() constexpr {
    std::array<std::uint8_t, 256> t{};
    for (int i = 0; i < 256; ++i) {
        std::uint8_t p = 0;
        for (int b = 0; b < 8; ++b) p ^= (i >> b) & 1;
        t[i] = p;
    }
    return t;
}();

inline std::uint8_t popcount_parity(std::uint8_t byte) {
    return kParityTable[byte];
}

template <std::int64_t size>
struct array_less {
    bool operator()(const std::array<std::uint8_t, size>& lhs,
                    const std::array<std::uint8_t, size>& rhs) const {
        for (std::int64_t i = 0; i < size; ++i) {
            if (lhs[i] < rhs[i]) return true;
            if (lhs[i] > rhs[i]) return false;
        }
        return false;
    }
};

inline bool get_bit(const std::uint8_t* data, std::int64_t index) {
    return (data[index >> 3] >> (index & 7)) & 1;
}

inline void set_bit(std::uint8_t* data, std::int64_t index, bool value) {
    if (value)
        data[index >> 3] |= (1 << (index & 7));
    else
        data[index >> 3] &= ~(1 << (index & 7));
}

template <std::int64_t n_qubytes>
std::uint8_t jw_parity(const std::uint8_t* config, std::int64_t site) {
    std::uint8_t p = 0;
    std::int64_t bi = static_cast<std::int64_t>(site >> 3);
    for (std::int64_t b = 0; b < bi; ++b)
        p ^= popcount_parity(config[b]);
    std::uint8_t mask = static_cast<std::uint8_t>((1 << (site & 7)) - 1);
    p ^= popcount_parity(config[bi] & mask);
    return p;
}

template <std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut, bool Forward>
std::pair<bool, bool> apply_operators(
    std::array<std::uint8_t, n_qubytes>& config,
    std::int64_t term_index,
    const std::array<std::int16_t, max_op_number>* site,
    const std::array<std::uint8_t, max_op_number>* kind)
{
    static_assert(particle_cut == 1 || particle_cut == 2);
    bool success = true;
    bool parity = false;

    if constexpr (Forward) {
        for (std::int64_t i = max_op_number; i-- > 0;) {
            std::uint8_t k = kind[term_index][i];
            if (k == 2) continue;
            std::int16_t s = site[term_index][i];
            if (get_bit(config.data(), s) == k) { success = false; break; }
            set_bit(config.data(), s, k);
            if constexpr (particle_cut == 1)
                parity ^= jw_parity<n_qubytes>(config.data(), s);
        }
    } else {
        for (std::int64_t i = 0; i < max_op_number; ++i) {
            std::uint8_t k = kind[term_index][i];
            if (k == 2) continue;
            std::int16_t s = site[term_index][i];
            bool target = 1 - k;
            if (get_bit(config.data(), s) == target) { success = false; break; }
            set_bit(config.data(), s, target);
            if constexpr (particle_cut == 1)
                parity ^= jw_parity<n_qubytes>(config.data(), s);
        }
    }
    return {success, parity};
}

template <std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
auto sort_configs(
    const torch::Tensor& configs,
    torch::Tensor& sort_idx)
{
    using Config = std::array<std::uint8_t, n_qubytes>;
    std::int64_t n = configs.size(0);
    sort_idx = torch::arange(n, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto* idx_ptr = sort_idx.data_ptr<std::int64_t>();
    const auto* cfg_ptr = reinterpret_cast<const Config*>(configs.data_ptr());
    std::sort(idx_ptr, idx_ptr + n, [cfg_ptr](std::int64_t a, std::int64_t b) {
        return array_less<n_qubytes>()(cfg_ptr[a], cfg_ptr[b]);
    });
    return configs.index({sort_idx});
}

template <std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
auto apply_within_interface(
    const torch::Tensor& configs_i,
    const torch::Tensor& psi_i,
    const torch::Tensor& configs_j,
    const torch::Tensor& site,
    const torch::Tensor& kind,
    const torch::Tensor& coef,
    bool configs_i_sorted,
    bool configs_j_sorted,
    int direction) -> torch::Tensor
{
    using Config = std::array<std::uint8_t, n_qubytes>;
    using Coef2 = std::array<double, 2>;

    std::int64_t batch_i = configs_i.size(0);
    std::int64_t batch_j = configs_j.size(0);
    std::int64_t term_number = site.size(0);

    const auto* site_ptr = reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr());
    const auto* kind_ptr = reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr());
    const auto* coef_ptr = reinterpret_cast<const Coef2*>(coef.data_ptr());

    auto less = array_less<n_qubytes>();

    bool forward = (direction == 0);

    auto result_psi = torch::zeros({batch_j, 2},
        torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto* result_ptr = reinterpret_cast<Coef2*>(result_psi.data_ptr());

    if (forward) {
        torch::Tensor sort_j_idx;
        auto sorted_j = configs_j_sorted
            ? configs_j
            : sort_configs<max_op_number, n_qubytes, particle_cut>(configs_j, sort_j_idx);

        const auto* sorted_j_ptr = reinterpret_cast<const Config*>(sorted_j.data_ptr());
        const auto* ci_ptr = reinterpret_cast<const Config*>(configs_i.data_ptr());
        const auto* pi_ptr = reinterpret_cast<const Coef2*>(psi_i.data_ptr());

        for (std::int64_t t = 0; t < term_number; ++t) {
            for (std::int64_t bi = 0; bi < batch_i; ++bi) {
                Config c = ci_ptr[bi];
                auto [ok, parity] = apply_operators<max_op_number, n_qubytes, particle_cut, true>(
                    c, t, site_ptr, kind_ptr);
                if (!ok) continue;
                std::int64_t lo = 0, hi = batch_j - 1;
                bool found = false;
                std::int64_t mid = 0;
                while (lo <= hi) {
                    mid = (lo + hi) / 2;
                    if (less(c, sorted_j_ptr[mid])) hi = mid - 1;
                    else if (less(sorted_j_ptr[mid], c)) lo = mid + 1;
                    else { found = true; break; }
                }
                if (!found) continue;
                double sign = parity ? -1.0 : 1.0;
                result_ptr[mid][0] += sign * (coef_ptr[t][0] * pi_ptr[bi][0] - coef_ptr[t][1] * pi_ptr[bi][1]);
                result_ptr[mid][1] += sign * (coef_ptr[t][0] * pi_ptr[bi][1] + coef_ptr[t][1] * pi_ptr[bi][0]);
            }
        }

        if (!configs_j_sorted) {
            auto unsorted = torch::zeros_like(result_psi);
            unsorted.index_put_({sort_j_idx}, result_psi);
            return unsorted;
        }
        return result_psi;

    } else {
        torch::Tensor sort_i_idx;
        auto sorted_i = configs_i_sorted
            ? configs_i
            : sort_configs<max_op_number, n_qubytes, particle_cut>(configs_i, sort_i_idx);
        torch::Tensor sorted_psi_i;
        if (configs_i_sorted) {
            sorted_psi_i = psi_i;
        } else {
            sorted_psi_i = torch::zeros({batch_i, 2},
                torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
            sorted_psi_i.index_put_({sort_i_idx}, psi_i);
        }

        const auto* sorted_i_ptr = reinterpret_cast<const Config*>(sorted_i.data_ptr());
        const auto* sorted_pi_ptr = reinterpret_cast<const Coef2*>(sorted_psi_i.data_ptr());
        const auto* cj_ptr = reinterpret_cast<const Config*>(configs_j.data_ptr());

        for (std::int64_t t = 0; t < term_number; ++t) {
            for (std::int64_t bj = 0; bj < batch_j; ++bj) {
                Config c = cj_ptr[bj];
                auto [ok, parity] = apply_operators<max_op_number, n_qubytes, particle_cut, false>(
                    c, t, site_ptr, kind_ptr);
                if (!ok) continue;
                std::int64_t lo = 0, hi = batch_i - 1;
                bool found = false;
                std::int64_t mid = 0;
                while (lo <= hi) {
                    mid = (lo + hi) / 2;
                    if (less(c, sorted_i_ptr[mid])) hi = mid - 1;
                    else if (less(sorted_i_ptr[mid], c)) lo = mid + 1;
                    else { found = true; break; }
                }
                if (!found) continue;
                double sign = parity ? -1.0 : 1.0;
                result_ptr[bj][0] += sign * (coef_ptr[t][0] * sorted_pi_ptr[mid][0] - coef_ptr[t][1] * sorted_pi_ptr[mid][1]);
                result_ptr[bj][1] += sign * (coef_ptr[t][0] * sorted_pi_ptr[mid][1] + coef_ptr[t][1] * sorted_pi_ptr[mid][0]);
            }
        }
        return result_psi;
    }
}

}  // namespace

#ifndef MAX_OP_NUMBER
#define MAX_OP_NUMBER 0
#endif
#ifndef N_QUBYTES
#define N_QUBYTES 0
#endif
#ifndef PARTICLE_CUT
#define PARTICLE_CUT 0
#endif

#if N_QUBYTES != 0
#define QMP_LIBRARY_HELPER(mo, nq, pc) qmp_hamiltonian_##mo##_##nq##_##pc
#define QMP_LIBRARY(mo, nq, pc) QMP_LIBRARY_HELPER(mo, nq, pc)
TORCH_LIBRARY_IMPL(QMP_LIBRARY(MAX_OP_NUMBER, N_QUBYTES, PARTICLE_CUT), CPU, m) {
    m.impl("apply_within_subspace_in_double_side",
           apply_within_interface<MAX_OP_NUMBER, N_QUBYTES, PARTICLE_CUT>);
}
#undef QMP_LIBRARY
#undef QMP_LIBRARY_HELPER
#endif
```

- [ ] **Step 1:** Write `_hamiltonian_cpu.cpp`
- [ ] **Step 2:** Commit `feat: add CPU backend for apply_within`

---

### Task 6: Run tests and debug

- [ ] **Step 1:** `PYTHONPATH=src python -m pytest tests/test_hamiltonian.py -v`
- [ ] **Step 2:** Fix any compilation/runtime issues
- [ ] **Step 3:** Iterate until all tests pass
- [ ] **Step 4:** Commit `fix: correct apply_within implementation`

---

### Task 7: CUDA backend stub

**Files:** Create `src/qmp/hamiltonian/_hamiltonian_cuda.cu`

Same as CPU backend but with `TORCH_LIBRARY_IMPL(..., CUDA, m)`. Use `__global__` kernel with 2D grid, `atomicAdd` for accumulation. (If no CUDA available, skip.)

- [ ] **Step 1:** Write stub/implementation
- [ ] **Step 2:** Commit

---

### Task 8: Final verification

- [ ] **Step 1:** `PYTHONPATH=src python -m pytest tests/test_hamiltonian.py -v -s` → all PASS
- [ ] **Step 2:** Verify deprecation: `python -W all -c "..."` → DeprecationWarning
- [ ] **Step 3:** Commit
