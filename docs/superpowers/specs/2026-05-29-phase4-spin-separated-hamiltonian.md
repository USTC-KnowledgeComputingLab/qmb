# Phase 4: `SpinSeparatedHamiltonian`

**Date**: 2026-05-29
**Status**: approved

## Goal

Implement `SpinSeparatedHamiltonian` — a new Hamiltonian type that internally separates spin-up and spin-down degrees of freedom. Input/output configs use interleaved order (0↑,0↓,1↑,1↓,...), consistent with `Hamiltonian`. Internally separated into up-block (even orbital indices) and down-block (odd orbital indices). No code reuse from `Hamiltonian`.

## Decisions

| Decision | Value |
|----------|-------|
| Code reuse | None. New files from scratch |
| Python/C++ boundary | Python is thin wrapper. C++ handles config separation via custom bit-permutation (not standard libtorch ops) |
| Config ordering | External: interleaved. Internal: block-separated |
| JW sign (cross-spin) | Cancels ONLY if same-spin operators are contiguous in each term. Standard chemistry Hamiltonians satisfy this naturally |
| Operator name | `spin_separated_apply_within_subspace_in_double_side` |
| Module naming | `qmp_spin_separated_hamiltonian_{NQ_UP}_{PC}_{MO}` |
| CPU backend | New four-layer kernel (not shared with `Hamiltonian`) |
| CUDA backend | New four-layer kernel, same patterns (CUDAGuard, stream, thrust) |

## Algorithm: Config Bit Separation

Interleaved configs (bits: 0↑,0↓,1↑,1↓,2↑,2↓,...) separated into up_block (bits: 0↑,2↑,4↑,...) and down_block (bits: 1↓,3↓,5↓,...). This is a bit-level gather operation: for an n-qubyte config, collect bits at even positions into the first ceil(n/16) bytes, odd positions into the remaining bytes. Done in C++ via a per-config permutation loop (not libtorch indexing — bit manipulation is faster).

## Algorithm: JW Sign Cancellation Condition

In block order, for a term with un-interleaved same-spin operators, the JW parity contribution from up_spin to down_spin is `parity(up_config)`. Since down operators within a single term always come in even numbers (creation+annihilation pairs), and the up_config is NOT modified during down-operator application (all up ops are applied later), the `parity(up_config)` XORs to zero.

This REQUIRES that same-spin operators be contiguous within each term tuple (Hubbard, Heisenberg, etc. satisfy this naturally). If operators are interleaved (up, down, up, down), the cancellation fails.

## Python API

```python
class SpinSeparatedHamiltonian:
    def __init__(
        self,
        hamiltonian: dict,
        *,
        kind: Literal["fermi", "bose2"],
        n_up: int,
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

## Architecture

```
_spin_separated_hamiltonian.cpp      ← prepare + TORCH_LIBRARY_FRAGMENT
_spin_separated_hamiltonian_cpu.cpp  ← CPU backend (config sep + up kernel + down kernel + accumulate)
_spin_separated_hamiltonian_cuda.cu  ← CUDA backend
```

### C++ Interface Layer (host, `TORCH_LIBRARY_IMPL`)

1. **Separate configs**: Bit-permutation loop extracts up_block and down_block from interleaved input
2. **For each term**: Determine if up-only, down-only, or mixed (both spins)
3. **Apply up operators**: standard kernel on up_block configs (JW within up block only)
4. **Apply down operators**: standard kernel on down_block configs (JW within down block only)
5. **Accumulate**: Cross-product the up and down contributions using binary search in separated destination configs

### Bit Separation (host helper function)

```cpp
// Extract even bits (0,2,4,...) from interleaved config → up_block config
// Extract odd bits (1,3,5,...) → down_block config
void separate_configs(
    const uint8_t* interleaved, int n_qubytes,
    uint8_t* up_block, int n_qubytes_up,
    uint8_t* down_block, int n_qubytes_down);
```

Uses per-byte bit-level shuffling (shift + mask) for performance.

## Tests

- 2-site Hubbard: compare `SpinSeparatedHamiltonian` results with `Hamiltonian` reference
- 4-site Hubbard: larger system, same comparison
- Forward == backward
- Boson mode
- Complex coefficients

## Files

```
src/qmp/hamiltonian/
├── _spin_separated_hamiltonian.py
├── _spin_separated_hamiltonian.cpp
├── _spin_separated_hamiltonian_cpu.cpp
└── _spin_separated_hamiltonian_cuda.cu
```

## Out of Scope

- find_relative, list_relative, diagonal_term
- Matrix-form psi representation (v1 uses 1D vector like `Hamiltonian`)
- Performance optimizations leveraging the separated representation
