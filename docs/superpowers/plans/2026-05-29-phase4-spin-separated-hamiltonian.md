# SpinSeparatedHamiltonian Implementation Plan

> **Status**: in progress

**Goal:** Create `SpinSeparatedHamiltonian` — new module that internally separates up/down spin degrees of freedom. Same external API as `Hamiltonian`, same results, new code.

**Architecture:** 3 new files (no code reuse). C++ backend receives interleaved configs, applies bit-permutation to separate into up/down blocks, runs standard kernel logic on each block, accumulates combined results.

---

### Task 0: Python class skeleton

**Files:** Create `src/qmp/hamiltonian/_spin_separated_hamiltonian.py`

Thin wrapper identical in style to `_hamiltonian.py`:
- `__init__` parses `n_up`, `max_op_number`, `devices`, `kind`
- `_load_module` uses `device_type` to pick `.cpp` / `.cpp`+`.cu` sources
- `_prepare` calls C++ prepare function
- `apply_within_subspace_in_double_side` calls C++ operator with direction flag

### Task 1: C++ declaration module

**Files:** Create `src/qmp/hamiltonian/_spin_separated_hamiltonian.cpp`

- `prepare` function (same pattern as `_hamiltonian.cpp`)
- `TORCH_LIBRARY_FRAGMENT` declaring `spin_separated_apply_within_subspace_in_double_side`
- Operator signature: same as `Hamiltonian`'s operator, but includes `n_up` parameter so the kernel knows the split point

### Task 2: CPU backend — bit separation + kernel

**Files:** Create `src/qmp/hamiltonian/_spin_separated_hamiltonian_cpu.cpp`

Key components:
1. **`separate_configs`**: Bit-permutation function that extracts even/odd bits from interleaved config
2. **`combine_configs`**: Inverse — merges up_block and down_block back into interleaved
3. **`hamiltonian_apply_kernel`**: Same logic as `_hamiltonian_cpu.cpp`, reused in-code (not as shared header)
4. **`apply_within_subspace_in_double_side_kernel`**: Per (term, batch) work unit
5. **`apply_within_subspace_in_double_side_kernel_interface`**: Loop over term × batch
6. **`apply_within_subspace_in_double_side_interface`**: PyTorch integration — separates configs, sorts, calls kernel, combines results
7. **`TORCH_LIBRARY_IMPL(CPU)`**

### Task 3: CUDA backend

**Files:** Create `src/qmp/hamiltonian/_spin_separated_hamiltonian_cuda.cu`

Same pattern as CPU backend, with `__global__` bit-permutation preprocessing kernel (runs once per call, not per term×batch). Stream-aware thrust sort. `TORCH_LIBRARY_IMPL(CUDA)`.

### Task 4: Tests

**Files:** Modify `tests/test_hamiltonian.py`

Add `TestSpinSeparated` class: construct both `Hamiltonian` and `SpinSeparatedHamiltonian` with same Hamiltonian dict, verify identical results on Hubbard model (2-site, 4-site).

### Task 5: Update docs

Update Hamiltonian `AGENTS.md` with spin separation section. Update spec/plan status.

### Task 6: Squash and finalize
