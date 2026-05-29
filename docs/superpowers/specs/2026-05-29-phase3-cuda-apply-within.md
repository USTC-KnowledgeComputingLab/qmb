# Phase 3: CUDA `apply_within_subspace_in_double_side`

**Date**: 2026-05-29
**Status**: implemented

## Goal

Port the `apply_within_subspace_in_double_side` operator to CUDA GPU, reusing the established four-layer architecture. No changes to CPU backend or Python wrapper.

## Decisions

| Decision | Value |
|----------|-------|
| Code layout | Single `_hamiltonian_cuda.cu`, self-contained, no shared headers |
| CPU backend | No modifications |
| Kernel grid | 2D: x=term_index, y=batch_index, block(1,1,1) |
| Config sorting | `thrust::sort_by_key` on device, stream-aware (`device.on(stream)`) |
| Result accumulation | `atomicAdd` (two double per complex) |
| Device management | `CUDAGuard` + `getCurrentCUDAStream` + `cudaDeviceProp` |
| Error checking | `AT_CUDA_CHECK(cudaStreamSynchronize(stream))` |
| Thread blocks | `dim3{1, maxThreadsPerBlock >> 1}` — x=1 term, y≈512 batch |
| Binary search | Same lexicographic comparator as CPU, in device memory |
| Forward/Backward | `bool` template parameter, `if constexpr` dispatch |
| JW parity | `std::popcount` (CPU) / `__popc` (CUDA) hardware instruction |

## Architecture

```
_hamiltonian_cuda.cu
├── popcount_parity / get_bit / set_bit / jw_parity  (__device__)
├── array_less                                        (__host__ __device__)
├── hamiltonian_apply_kernel                          (__device__)  — pure computation
├── apply_within_subspace_in_double_side_kernel       (__device__)  — per (term, batch)
├── apply_within_subspace_in_double_side_kernel_interface (__global__) — 2D grid
└── apply_within_subspace_in_double_side_interface    (host) — sort_configs_cuda + launch + unsort
```

## CUDA Kernel Flow

Forward mode:
1. `thrust::sort_by_key` sorts configs_j on device, keeping original indices
2. Launch `__global__` kernel with `grid(term_number, batch_i)`
3. Each thread: apply operators, binary search in sorted_j, `atomicAdd` result
4. `cudaDeviceSynchronize()`
5. Unsort result back using original indices

Backward mode:
1. `thrust::sort_by_key` sorts configs_i on device
2. Rearrange psi_i to match sorted order
3. Launch kernel with `grid(term_number, batch_j)` — iterate configs_j, binary search config_i
4. `cudaDeviceSynchronize()`
5. Return result (indexed by configs_j, no unsort needed)

## Tests

Mirror CPU test coverage on GPU:
- Basic fermion hopping (single, hermitian conjugate, Pauli exclusion)
- Multiple contributions to same target
- Complex coefficients
- Boson (no JW sign)
- Forward == backward
- Hubbard diagonal U term + hopping
- Edge cases: empty batches, identity, superset/subset, duplicates, unsorted, diagonal-only
- All CUDA results compared against CPU reference output

## Files

```
src/qmp/hamiltonian/
└── _hamiltonian_cuda.cu          ← complete rewrite

tests/
└── test_hamiltonian.py           ← append TestCUDA class
```

## Out of Scope

- find_relative, list_relative, diagonal_term (future phases)
- Multi-GPU execution
- Thread-block-level optimizations (shared memory, warp shuffle)
