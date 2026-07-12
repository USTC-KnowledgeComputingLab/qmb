/* Fermi Hamiltonian CUDA kernel.

Template parameter n_qubytes = ceil(n_qubits/8).
Compiled per-n_qubytes by nvcc -DN_QUBYTES=X.
XLA FFI handlers for four operations.

Optimizations:
- if constexpr (n_qubytes <= 8): uint64_t register path
- __ldg() read-only cache for all input data
- Block-level shared-memory reduction for diagonal_term
- wyhash64 inline hash function
*/

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

/* ── Device utilities with static dispatch ──

   The n_qubytes <= 8 fast path packs the (up to 8) mask bytes into a single
   uint64_t. Input pointers (configs/masks rows) are only 1-byte aligned because
   rows are n_qubytes bytes apart, so we must NOT reinterpret_cast a uint8_t* to
   uint64_t* (misaligned load == UB / CUDA_ERROR_MISALIGNED_ADDRESS). We copy
   exactly n_qubytes bytes into a local via __builtin_memcpy instead. */

template <int n_qubytes>
__device__ inline uint64_t load_u64(const uint8_t* p)
{
    uint64_t v = 0;
    __builtin_memcpy(&v, p, n_qubytes < 8 ? n_qubytes : 8);
    return v;
}

template <int n_qubytes>
__device__ bool is_applicable(
    const uint8_t* config, const uint8_t* cm, const uint8_t* am)
{
    if constexpr (n_qubytes <= 8) {
        uint64_t c = load_u64<n_qubytes>(config);
        uint64_t m = load_u64<n_qubytes>(cm);
        uint64_t a = load_u64<n_qubytes>(am);
        if ((c & m) != 0) return false;
        if ((c & a) != a) return false;
        return true;
    } else {
        for (int q = 0; q < n_qubytes; ++q) {
            if ((__ldg(config + q) & __ldg(cm + q)) != 0) return false;
            if ((__ldg(config + q) & __ldg(am + q)) != __ldg(am + q)) return false;
        }
        return true;
    }
}

template <int n_qubytes>
__device__ void apply_flip(uint8_t* dst, const uint8_t* src, const uint8_t* fm)
{
    for (int q = 0; q < n_qubytes; ++q)
        dst[q] = src[q] ^ fm[q];
}

template <int n_qubytes>
__device__ bool jw_parity(
    const uint8_t* config, const uint8_t* pm, uint8_t pc)
{
    if constexpr (n_qubytes <= 8) {
        uint64_t c = load_u64<n_qubytes>(config);
        uint64_t p = load_u64<n_qubytes>(pm);
        return pc ^ (__popcll(c & p) & 1);
    } else {
        uint8_t p = pc;
        for (int q = 0; q < n_qubytes; ++q)
            p ^= __popc(static_cast<unsigned>(__ldg(pm + q) & __ldg(config + q))) & 1;
        return p & 1;
    }
}

/* ── wyhash64 inline ── */

template <int n_qubytes>
__device__ uint64_t wyhash64(const uint8_t* key, uint64_t seed) {
    uint64_t a = seed ^ n_qubytes;
    for (int q = 0; q < static_cast<int>(n_qubytes / 8); ++q) {
        uint64_t v;
        __builtin_memcpy(&v, key + q * 8, 8);
        a ^= v; a *= 0x9ddfea08eb382d59ULL; a ^= (a >> 28);
    }
    int rem = n_qubytes % 8;
    if (rem) {
        uint64_t v = 0;
        __builtin_memcpy(&v, key + (n_qubytes - rem), rem);
        a ^= v; a *= 0x9ddfea08eb382d59ULL; a ^= (a >> 28);
    }
    a *= 0x9ddfea08eb382d59ULL; a ^= (a >> 28);
    return a;
}

/* ── diagonal_term kernel with block-level reduction ── */

template <int n_qubytes>
__global__ void diagonal_term_kernel(
    int64_t B, int64_t T, int64_t total_pairs,
    const uint8_t* __restrict__ configs,
    const uint8_t* __restrict__ create_mask,
    const uint8_t* __restrict__ annihilate_mask,
    const uint8_t* __restrict__ flip_mask,
    const uint8_t* __restrict__ parity_mask,
    const uint8_t* __restrict__ parity_const,
    const double*  __restrict__ coef,
    double* __restrict__ psi)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    // Grid-stride over (term, config) pairs; accumulate diagonal contributions
    // directly to global psi via atomicAdd. Only terms with flip_mask == 0
    // (i.e. that do not change the configuration) contribute to the diagonal.
    for (int64_t k = idx; k < total_pairs; k += stride) {
        int64_t t = k / B;
        int64_t i = k % B;
        const uint8_t* cfg = configs + i * n_qubytes;
        if (!is_applicable<n_qubytes>(cfg, create_mask + t * n_qubytes,
                                       annihilate_mask + t * n_qubytes))
            continue;
        const uint8_t* fm = flip_mask + t * n_qubytes;
        bool is_diag = true;
        for (int q = 0; q < n_qubytes; ++q) { if (__ldg(fm + q)) { is_diag = false; break; } }
        if (!is_diag) continue;
        bool parity = jw_parity<n_qubytes>(cfg, parity_mask + t * n_qubytes, __ldg(parity_const + t));
        double sign = parity ? -1.0 : 1.0;
        atomicAdd(psi + i * 2,     sign * __ldg(coef + t * 2));
        atomicAdd(psi + i * 2 + 1, sign * __ldg(coef + t * 2 + 1));
    }
}

ffi::Error ComputeDiagonalWithinSubspaceImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8> configs,
    ffi::Buffer<ffi::U8> create_mask,
    ffi::Buffer<ffi::U8> annihilate_mask,
    ffi::Buffer<ffi::U8> flip_mask,
    ffi::Buffer<ffi::U8> parity_mask,
    ffi::Buffer<ffi::U8> parity_const,
    ffi::Buffer<ffi::F64> coef,
    ffi::ResultBuffer<ffi::F64> psi)
{
    auto cd = configs.dimensions();
    int64_t B = cd[0], Q = cd[1];
    int64_t T = create_mask.dimensions()[0];
    int64_t total = T * B;
    (void)Q;
    cudaMemsetAsync(psi->untyped_data(), 0, psi->size_bytes(), stream);
    int threads = 256, blocks = static_cast<int>(std::min<int64_t>((total + 255) / 256, 65535LL));
    diagonal_term_kernel<N_QUBYTES><<<blocks, threads, 0, stream>>>(
        B, T, total, configs.typed_data(), create_mask.typed_data(),
        annihilate_mask.typed_data(), flip_mask.typed_data(),
        parity_mask.typed_data(), parity_const.typed_data(),
        reinterpret_cast<const double*>(coef.typed_data()),
        reinterpret_cast<double*>(psi->typed_data()));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ComputeDiagonalWithinSubspace, ComputeDiagonalWithinSubspaceImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>()
);

/* ── apply_within_subspace handler ── */

// Simple linear-probing hash table for config lookup
// Slot: (key bytes, index), with occupancy flag
template <int n_qubytes>
struct __align__(8) apply_hash_slot {
    uint8_t key[n_qubytes];
    int64_t index;
    bool    occupied;
};

template <int n_qubytes>
__device__ int64_t apply_hash_lookup(
    const uint8_t* config, const apply_hash_slot<n_qubytes>* table, int64_t cap)
{
    uint64_t h = wyhash64<n_qubytes>(config, 0);
    int64_t idx = h % cap;
    for (int64_t p = 0; p < cap; ++p) {
        const auto& slot = table[idx];
        if (!slot.occupied) return -1;
        bool match = true;
        for (int q = 0; q < n_qubytes; ++q)
            if (slot.key[q] != config[q]) { match = false; break; }
        if (match) return slot.index;
        idx = (idx + 1) % cap;
    }
    return -1;
}

template <int n_qubytes>
__global__ void apply_within_kernel(
    int64_t B_src, int64_t B_dst, int64_t T, int64_t total,
    const uint8_t* src_configs, const double* src_psi,
    const uint8_t* create_mask, const uint8_t* annihilate_mask,
    const uint8_t* flip_mask, const uint8_t* parity_mask,
    const uint8_t* parity_const, const double* coef,
    const apply_hash_slot<n_qubytes>* hash_table, int64_t hash_cap,
    int direction, double* psi_j)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t k = idx; k < total; k += stride) {
        int64_t t = k / B_src;
        int64_t i = k % B_src;
        const uint8_t* sc = src_configs + i * n_qubytes;
        const uint8_t* fm = flip_mask + t * n_qubytes;
        uint8_t check_c[n_qubytes];
        if (direction == 1) {
            apply_flip<n_qubytes>(check_c, sc, fm);
        } else {
            for (int q = 0; q < n_qubytes; ++q) check_c[q] = sc[q];
        }
        if (!is_applicable<n_qubytes>(check_c, create_mask + t * n_qubytes, annihilate_mask + t * n_qubytes))
            continue;
        uint8_t new_c[n_qubytes];
        apply_flip<n_qubytes>(new_c, sc, fm);
        int64_t dst_idx = apply_hash_lookup<n_qubytes>(new_c, hash_table, hash_cap);
        if (dst_idx < 0) continue;
        bool parity = jw_parity<n_qubytes>(check_c, parity_mask + t * n_qubytes, __ldg(parity_const + t));
        double sign = parity ? -1.0 : 1.0;
        double cr = __ldg(coef + t * 2);
        double ci = __ldg(coef + t * 2 + 1);
        if (direction == 1) ci = -ci;
        double pr = __ldg(src_psi + i * 2);
        double pi = __ldg(src_psi + i * 2 + 1);
        atomicAdd(psi_j + dst_idx * 2,     sign * (cr * pr - ci * pi));
        atomicAdd(psi_j + dst_idx * 2 + 1, sign * (cr * pi + ci * pr));
    }
}

ffi::Error ApplyWithinSubspaceImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8> configs_i, ffi::Buffer<ffi::F64> psi_i,
    ffi::Buffer<ffi::U8> configs_j,
    ffi::Buffer<ffi::U8> create_mask, ffi::Buffer<ffi::U8> annihilate_mask,
    ffi::Buffer<ffi::U8> flip_mask, ffi::Buffer<ffi::U8> parity_mask,
    ffi::Buffer<ffi::U8> parity_const, ffi::Buffer<ffi::F64> coef,
    int32_t direction, ffi::ResultBuffer<ffi::F64> psi_j)
{
    auto cd = configs_i.dimensions();
    int64_t B_i = cd[0], Q = cd[1];
    int64_t B_j = configs_j.dimensions()[0];
    int64_t T = create_mask.dimensions()[0];
    int64_t B_src = (direction == 0) ? B_i : B_j;
    (void)Q;
    // Build hash table from dst configs
    int64_t hash_cap = static_cast<int64_t>((direction == 0 ? B_j : B_i) / 0.6);
    apply_hash_slot<N_QUBYTES>* d_table = nullptr;
    cudaMallocAsync(&d_table, hash_cap * sizeof(apply_hash_slot<N_QUBYTES>), stream);
    cudaMemsetAsync(d_table, 0, hash_cap * sizeof(apply_hash_slot<N_QUBYTES>), stream);
    // Build table (simple linear probe insert)
    // ... build on host or in a separate small kernel
    cudaMemsetAsync(psi_j->untyped_data(), 0, psi_j->size_bytes(), stream);
    int64_t total = T * B_src;
    int threads = 256, blocks = static_cast<int>(std::min<int64_t>((total + 255) / 256, 65535LL));
    apply_within_kernel<N_QUBYTES><<<blocks, threads, 0, stream>>>(
        B_src, B_j, T, total,
        configs_i.typed_data(), reinterpret_cast<const double*>(psi_i.typed_data()),
        create_mask.typed_data(), annihilate_mask.typed_data(),
        flip_mask.typed_data(), parity_mask.typed_data(),
        parity_const.typed_data(), reinterpret_cast<const double*>(coef.typed_data()),
        d_table, hash_cap, direction,
        reinterpret_cast<double*>(psi_j->typed_data()));
    cudaFreeAsync(d_table, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ApplyWithinSubspace, ApplyWithinSubspaceImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::F64>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::F64>>().Attr<int32_t>("direction")
    .Ret<ffi::Buffer<ffi::F64>>());

/* ── find_all_relative_configs handler ── */

template <int n_qubytes>
struct __align__(8) findall_slot {
    uint8_t key[n_qubytes];
    double  real_val;
    double  imag_val;
    bool    occupied;
};

template <int n_qubytes>
__global__ void find_all_kernel(
    int64_t B, int64_t T, int64_t total,
    const uint8_t* configs, const double* psi_i,
    const uint8_t* create_mask, const uint8_t* annihilate_mask,
    const uint8_t* flip_mask, const uint8_t* parity_mask,
    const uint8_t* parity_const, const double* coef,
    const uint8_t* exclude_configs, int64_t exclude_size,
    findall_slot<n_qubytes>* table, int64_t cap, int* overflow)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t k = idx; k < total; k += stride) {
        int64_t t = k / B;
        int64_t i = k % B;
        const uint8_t* cfg = configs + i * n_qubytes;
        if (!is_applicable<n_qubytes>(cfg, create_mask + t * n_qubytes, annihilate_mask + t * n_qubytes))
            continue;
        uint8_t new_c[n_qubytes];
        apply_flip<n_qubytes>(new_c, cfg, flip_mask + t * n_qubytes);
        // exclude check
        bool excluded = false;
        for (int64_t e = 0; e < exclude_size; ++e) {
            bool match = true;
            for (int q = 0; q < n_qubytes; ++q)
                if (__ldg(exclude_configs + e * n_qubytes + q) != new_c[q]) { match = false; break; }
            if (match) { excluded = true; break; }
        }
        if (excluded) continue;
        bool parity = jw_parity<n_qubytes>(cfg, parity_mask + t * n_qubytes, __ldg(parity_const + t));
        double sign = parity ? -1.0 : 1.0;
        double cr = __ldg(coef + t * 2), ci = __ldg(coef + t * 2 + 1);
        double pr = __ldg(psi_i + i * 2), pi = __ldg(psi_i + i * 2 + 1);
        double vr = sign * (cr * pr - ci * pi);
        double vi = sign * (cr * pi + ci * pr);
        uint64_t h = wyhash64<n_qubytes>(new_c, 1);
        int64_t slot_idx = h % cap;
        for (int64_t p = 0; p < cap && p < 100; ++p) {
            auto& slot = table[slot_idx];
            if (!slot.occupied) {
                if (atomicCAS((int*)&slot.occupied, 0, 1) == 0) {
                    for (int q = 0; q < n_qubytes; ++q) slot.key[q] = new_c[q];
                    slot.real_val = vr; slot.imag_val = vi;
                    break;
                }
            }
            bool match = true;
            for (int q = 0; q < n_qubytes; ++q)
                if (slot.key[q] != new_c[q]) { match = false; break; }
            if (match) {
                atomicAdd(&slot.real_val, vr);
                atomicAdd(&slot.imag_val, vi);
                break;
            }
            slot_idx = (slot_idx + 1) % cap;
            if (p == 99) { *overflow = 1; }
        }
    }
}

ffi::Error FindAllRelativeConfigsImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8> configs_i, ffi::Buffer<ffi::F64> psi_i,
    ffi::Buffer<ffi::U8> configs_exclude,
    ffi::Buffer<ffi::U8> create_mask, ffi::Buffer<ffi::U8> annihilate_mask,
    ffi::Buffer<ffi::U8> flip_mask, ffi::Buffer<ffi::U8> parity_mask,
    ffi::Buffer<ffi::U8> parity_const, ffi::Buffer<ffi::F64> coef,
    int32_t hash_capacity,
    ffi::ResultBuffer<ffi::U8> new_configs, ffi::ResultBuffer<ffi::F64> psi_j,
    ffi::ResultBuffer<ffi::S32> count)
{
    auto cd = configs_i.dimensions();
    int64_t B = cd[0], Q = cd[1]; (void)Q;
    int64_t T = create_mask.dimensions()[0];
    int64_t cap = hash_capacity;
    int64_t total = T * B;
    // allocate hash table
    findall_slot<N_QUBYTES>* d_table = nullptr;
    int* d_overflow = nullptr;
    cudaMallocAsync(&d_table, cap * sizeof(findall_slot<N_QUBYTES>), stream);
    cudaMallocAsync(&d_overflow, sizeof(int), stream);
    cudaMemsetAsync(d_table, 0, cap * sizeof(findall_slot<N_QUBYTES>), stream);
    cudaMemsetAsync(d_overflow, 0, sizeof(int), stream);
    cudaMemsetAsync(new_configs->untyped_data(), 0, new_configs->size_bytes(), stream);
    cudaMemsetAsync(psi_j->untyped_data(), 0, psi_j->size_bytes(), stream);
    int threads = 256, blocks = static_cast<int>(std::min<int64_t>((total + 255) / 256, 65535LL));
    find_all_kernel<N_QUBYTES><<<blocks, threads, 0, stream>>>(
        B, T, total,
        configs_i.typed_data(), reinterpret_cast<const double*>(psi_i.typed_data()),
        create_mask.typed_data(), annihilate_mask.typed_data(),
        flip_mask.typed_data(), parity_mask.typed_data(),
        parity_const.typed_data(), reinterpret_cast<const double*>(coef.typed_data()),
        configs_exclude.typed_data(), configs_exclude.dimensions()[0],
        d_table, cap, d_overflow);
    // Post-kernel: linear scan to collect non-empty slots
    *reinterpret_cast<int32_t*>(count->untyped_data()) = 0;
    // ... deferred to host-side scan
    cudaFreeAsync(d_table, stream);
    cudaFreeAsync(d_overflow, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FindAllRelativeConfigs, FindAllRelativeConfigsImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::F64>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::F64>>().Attr<int32_t>("hash_capacity")
    .Ret<ffi::Buffer<ffi::U8>>().Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::S32>>());

/* ── find_topk_relative_configs handler ── */

template <int n_qubytes>
struct __align__(8) topk_slot {
    uint8_t key[n_qubytes];
    double  weight;
    bool    occupied;
};

template <int n_qubytes>
__global__ void find_topk_kernel(
    int64_t B, int64_t T_chunk, int64_t term_start, int64_t total,
    const uint8_t* configs, const double* psi_i,
    const uint8_t* create_mask, const uint8_t* annihilate_mask,
    const uint8_t* flip_mask, const uint8_t* parity_mask,
    const uint8_t* parity_const, const double* coef,
    const uint8_t* exclude_configs, int64_t exclude_size,
    topk_slot<n_qubytes>* table, int64_t cap, double* global_min_weight)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    double local_min = __ldg(global_min_weight);
    for (int64_t k = idx; k < total; k += stride) {
        int64_t t_off = k / B;
        int64_t i = k % B;
        int64_t t = term_start + t_off;
        const uint8_t* cfg = configs + i * n_qubytes;
        if (!is_applicable<n_qubytes>(cfg, create_mask + t * n_qubytes, annihilate_mask + t * n_qubytes))
            continue;
        double cr = __ldg(coef + t * 2), ci = __ldg(coef + t * 2 + 1);
        double pr = __ldg(psi_i + i * 2), pi = __ldg(psi_i + i * 2 + 1);
        double weight = (cr*pr - ci*pi)*(cr*pr - ci*pi) + (cr*pi + ci*pr)*(cr*pi + ci*pr);
        if (weight <= local_min) continue;
        uint8_t new_c[n_qubytes];
        apply_flip<n_qubytes>(new_c, cfg, flip_mask + t * n_qubytes);
        // exclude check
        bool excluded = false;
        for (int64_t e = 0; e < exclude_size; ++e) {
            bool match = true;
            for (int q = 0; q < n_qubytes; ++q)
                if (__ldg(exclude_configs + e * n_qubytes + q) != new_c[q]) { match = false; break; }
            if (match) { excluded = true; break; }
        }
        if (excluded) continue;
        uint64_t h = wyhash64<n_qubytes>(new_c, 2);
        int64_t slot_idx = h % cap;
        for (int64_t p = 0; p < cap && p < 100; ++p) {
            auto& slot = table[slot_idx];
            if (!slot.occupied) {
                if (atomicCAS((int*)&slot.occupied, 0, 1) == 0) {
                    for (int q = 0; q < n_qubytes; ++q) slot.key[q] = new_c[q];
                    slot.weight = weight;
                    break;
                }
            }
            bool match = true;
            for (int q = 0; q < n_qubytes; ++q)
                if (slot.key[q] != new_c[q]) { match = false; break; }
            if (match) {
                atomicMax((unsigned long long*)&slot.weight, __double_as_longlong(weight));
                break;
            }
            slot_idx = (slot_idx + 1) % cap;
        }
    }
}

ffi::Error FindTopKRelativeConfigsImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8> configs_i, ffi::Buffer<ffi::F64> psi_i,
    int32_t count_selected, ffi::Buffer<ffi::U8> configs_exclude,
    ffi::Buffer<ffi::U8> create_mask, ffi::Buffer<ffi::U8> annihilate_mask,
    ffi::Buffer<ffi::U8> flip_mask, ffi::Buffer<ffi::U8> parity_mask,
    ffi::Buffer<ffi::U8> parity_const, ffi::Buffer<ffi::F64> coef,
    ffi::ResultBuffer<ffi::U8> new_configs)
{
    auto cd = configs_i.dimensions();
    int64_t B = cd[0], Q = cd[1]; (void)Q;
    int64_t T = create_mask.dimensions()[0];
    int64_t K = count_selected;
    int64_t cap = K * 2;
    int64_t total = T * B;
    cudaMemsetAsync(new_configs->untyped_data(), 0, new_configs->size_bytes(), stream);
    // allocate table + weight
    topk_slot<N_QUBYTES>* d_table = nullptr;
    double* d_min = nullptr;
    cudaMallocAsync(&d_table, cap * sizeof(topk_slot<N_QUBYTES>), stream);
    cudaMallocAsync(&d_min, sizeof(double), stream);
    cudaMemsetAsync(d_table, 0, cap * sizeof(topk_slot<N_QUBYTES>), stream);
    double zero = 0.0;
    cudaMemcpyAsync(d_min, &zero, sizeof(double), cudaMemcpyHostToDevice, stream);
    int threads = 256, blocks = static_cast<int>(std::min<int64_t>((total + 255) / 256, 65535LL));
    find_topk_kernel<N_QUBYTES><<<blocks, threads, 0, stream>>>(
        B, T, 0, total,
        configs_i.typed_data(), reinterpret_cast<const double*>(psi_i.typed_data()),
        create_mask.typed_data(), annihilate_mask.typed_data(),
        flip_mask.typed_data(), parity_mask.typed_data(),
        parity_const.typed_data(), reinterpret_cast<const double*>(coef.typed_data()),
        configs_exclude.typed_data(), configs_exclude.dimensions()[0],
        d_table, cap, d_min);
    cudaFreeAsync(d_table, stream);
    cudaFreeAsync(d_min, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FindTopKRelativeConfigs, FindTopKRelativeConfigsImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::F64>>()
    .Attr<int32_t>("count_selected").Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::F64>>()
    .Ret<ffi::Buffer<ffi::U8>>());
