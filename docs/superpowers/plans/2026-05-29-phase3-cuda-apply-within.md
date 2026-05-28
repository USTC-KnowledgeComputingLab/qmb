# CUDA apply_within_subspace_in_double_side Implementation Plan

> **Status**: completed

**Goal:** Rewrite `_hamiltonian_cuda.cu` with full GPU implementation, mirror CPU tests on CUDA.

**Architecture:** Single self-contained `.cu` file, four-layer design, `thrust::sort_by_key` for config sorting, `atomicAdd` for accumulation. No CPU code changes.

---

### Task 0: Write CUDA backend `_hamiltonian_cuda.cu`

**Files:** Modify `src/qmp/hamiltonian/_hamiltonian_cuda.cu`

Complete rewrite with all utilities and four layers inline. Key code:

```cpp
#include <torch/extension.h>
#include <array>
#include <algorithm>
#include <cstdint>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace {

__device__ constexpr std::array<std::uint8_t, 256> kParityTable = {{
    // 256 precomputed parity values
}};

__device__ inline std::uint8_t popcount_parity(...) { ... }
__device__ inline bool get_bit(...) { ... }
__device__ inline void set_bit(...) { ... }

template <...> struct array_less { __host__ __device__ ... };

template <...> __device__ std::uint8_t jw_parity(...) { ... }

template <..., bool forward>
__device__ std::pair<bool, bool> hamiltonian_apply_kernel(...) { ... }

template <..., bool forward>
__device__ void apply_within_subspace_in_double_side_kernel(...) {
    // binary search + atomicAdd
}

template <..., bool forward>
__global__ void apply_within_subspace_in_double_side_kernel_interface(...) {
    int64_t t = blockIdx.x, b = blockIdx.y;
    if (t >= term_number || b >= src_batch_size) return;
    apply_within_subspace_in_double_side_kernel<..., forward>(t, b, ...);
}

template <...>
auto apply_within_subspace_in_double_side_interface(tensors..., bool sorted..., int64_t direction) -> Tensor {
    // forward: thrust::sort_by_key configs_j, launch kernel, unsort
    // backward: thrust::sort_by_key configs_i, rearrange psi, launch kernel
}

}  // namespace

// macros + TORCH_LIBRARY_FRAGMENT + TORCH_LIBRARY_IMPL(CUDA)
```

- [ ] Write complete `_hamiltonian_cuda.cu`
- [ ] Clear cache, verify it compiles
- [ ] Commit

---

### Task 1: Add CUDA tests

**Files:** Modify `tests/test_hamiltonian.py`

Add `TestCUDA` class mirroring CPU tests, validating against CPU reference:

```python
class TestCUDA:
    # Helper
    def _cpu_ham(self, ham_dict, kind="fermi", max_op_number=4):
        return Hamiltonian(ham_dict, kind=kind, max_op_number=max_op_number, devices=["localhost:cpu:0"])
    def _cuda_ham(self, ham_dict, kind="fermi", max_op_number=4):
        return Hamiltonian(ham_dict, kind=kind, max_op_number=max_op_number, devices=["localhost:cuda:0"])

    def test_same_as_cpu(self): ...
    def test_forward_backward(self): ...
    def test_boson(self): ...
    def test_hubbard_hopping(self): ...
    def test_complex_coef(self): ...
    def test_identity(self): ...
    def test_empty_batch(self): ...
    def test_unsorted(self): ...
    def test_diagonal_only(self): ...
    # ~15 tests total
```

- [ ] Write `TestCUDA` class
- [ ] Run all tests: `uv run pytest tests/test_hamiltonian.py -v`
- [ ] Commit

---

### Task 2: Update docs

**Files:** Create spec, update plan status

- [ ] Verify spec matches implementation
- [ ] Update plan status to completed
- [ ] Commit

---

### Task 3: Cleanup and squash

- [ ] `uv run ruff check . && uv run ty check src/qmp tests`
- [ ] Squash into feature commit
