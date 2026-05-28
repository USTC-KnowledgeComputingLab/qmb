# Phase 2: `apply_within_subspace_in_double_side`

**Date**: 2026-05-28
**Status**: implemented

## Goal

Implement the Hamiltonian subsystem's `apply_within_subspace_in_double_side` operation — Jacobian-free H|ψ⟩ projection onto a specified configuration subspace, with CPU and CUDA backends via JIT compilation.

## Decisions

| Decision | Value |
|----------|-------|
| Python version | `>=3.13` (use `warnings.deprecated`) |
| Build macros | `N_QUBYTES`, `PARTICLE_CUT`, `MAX_OP_NUMBER` |
| Module naming | `qmp_hamiltonian_{NQ}_{PC}_{MO}` |
| Parameter ordering | `n_qubytes`, `particle_cut`, `max_op_number`, `forward` |
| Operator name | `apply_within_subspace_in_double_side` |
| Direction modes | `forward` (term×config_i→config_j), `backward` (term×config_j→config_i) |
| Devices | Required param, currently `["localhost:cuda:0"]` or `["localhost:cpu:0"]` |
| kind type | `Literal["fermi", "bose2"]` |
| JW sign precompute | `std::popcount` (C++20 `<bit>`) hardware instruction |
| Sorting hints | `configs_i_sorted`, `configs_j_sorted` — skip redundant sort when true |
| Forward/Backward | Template parameter (compile-time dispatch) |
| CPU backend | Reference implementation, serial correctness baseline |

## Python API

```python
class Hamiltonian:
    def __init__(
        self,
        hamiltonian: dict,
        *,
        kind: Literal["fermi", "bose2"],
        max_op_number: int,
        devices: list[str],
    )

    def apply_within_subspace_in_double_side(
        self,
        configs_i: Tensor,
        psi_i: Tensor,
        configs_j: Tensor,
        *,
        configs_i_sorted: bool = False,
        configs_j_sorted: bool = False,
        direction: Literal["forward", "backward"] = "forward",
    ) -> Tensor

    @warnings.deprecated("use apply_within_subspace_in_double_side")
    def apply_within(self, *args, **kwargs) -> Tensor
```

## JW Sign Precomputation

`std::popcount` (C++20 `<bit>`) computes the popcount parity in a single instruction: `std::popcount(byte) & 1`. Replaces the old O(site_index) per-bit loop with an O(n_qubytes) per-byte scan. The CUDA equivalent is `__popc`。

## Two Direction Modes

| Mode | Loop structure | Best when |
|------|---------------|-----------|
| `forward` | for term × config_i: apply → binary_search config_j | batch_i < batch_j |
| `backward` | for term × config_j: apply_inverse → binary_search config_i | batch_j < batch_i |

Backward: iterate operators left-to-right, invert kind (0↔1), same JW sign logic.

Both modes require sorted target arrays. `configs_i_sorted`/`configs_j_sorted` skip the sort step.

The direction is a `bool` template parameter (`forward`) in C++ for compile-time dispatch via `if constexpr`.

## C++ Kernel/Interface Separation

Four-layer architecture for clean CPU-to-CUDA migration:

1. `hamiltonian_apply_kernel` — pure computation (no PyTorch). Operates on raw `std::array` pointers.
2. `apply_within_subspace_in_double_side_kernel` — per (term, batch) work unit. Calls kernel, binary searches, accumulates.
3. `apply_within_subspace_in_double_side_kernel_interface` — host-side loop over term×batch.
4. `apply_within_subspace_in_double_side_interface` — PyTorch integration (tensor handling, sorting, pointer extraction).

All template parameters use `lower_case` naming and follow the standard ordering: `n_qubytes`, `particle_cut`, `max_op_number`, `forward`.

## Device Ownership

The `Hamiltonian` class owns a computation device, determined at init from `devices[0]`. Input tensors are moved to the Hamiltonian's device in each operation. Hamiltonian internals (`site`, `kind`, `coef`) are placed on this device at init. Never the reverse.

## Tests

Cover:
- 2-site fermion hopping (basic correctness, JW sign)
- 2-site boson hopping (no JW sign)
- Pauli exclusion
- Multiple terms hitting same target
- Forward == backward
- Deprecation warning
- Multi-device → NotImplementedError
- 4-site spinful Hubbard (multi-orbital, U term with 4 operators per term)

## Files

```
src/qmp/hamiltonian/
├── __init__.py
├── AGENTS.md
├── _hamiltonian.py
├── _hamiltonian.cpp
├── _hamiltonian_cpu.cpp
└── _hamiltonian_cuda.cu

tests/
└── test_hamiltonian.py
```

## Out of Scope

- find_relative, list_relative, diagonal_term (future phases)
- Multi-device/multi-GPU execution (raises NotImplementedError)
- OpenMP parallelism in CPU backend
