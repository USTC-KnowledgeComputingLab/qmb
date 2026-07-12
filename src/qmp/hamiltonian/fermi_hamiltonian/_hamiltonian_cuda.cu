/* Fermi Hamiltonian CUDA kernel.

Template parameter n_qubytes = ceil(n_qubits/8).
Compiled per-n_qubytes by nvcc -DN_QUBYTES=X.
XLA FFI handlers for four operations.

Implementation notes (see spec 3.2 for the full design, incl. planned items):
- if constexpr (n_qubytes <= 8): pack masks into a single uint64_t register path
  (loaded via __builtin_memcpy — config/mask rows are only 1-byte aligned, so a
  reinterpret_cast to uint64_t* would be a misaligned/UB load).
- inline wyhash64 + custom linear-probing hash tables (no cuCollections).
- __ldg() read-only cache: used for coef/psi/parity_const and the n_qubytes>8
  byte-loop path. The n_qubytes<=8 packed-uint64 fast path deliberately does NOT
  use __ldg: config/mask rows are only 1-byte aligned, so a 64-bit __ldg would
  be a misaligned load; a byte-wise __ldg would defeat the single-register load.
  (This is a design exclusion, not a pending optimization.)
- diagonal_term: currently accumulates directly to global psi via atomicAdd.
  Shared-memory block reduction (spec 3.2.1) is planned, not yet implemented.
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

/* atomic max on double via CAS loop (valid for all doubles, unlike a raw
   bit-pattern atomicMax which only works for non-negative values). */
__device__ inline void atomic_max_double(double* addr, double val)
{
    unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *a, assumed;
    do {
        double cur = __longlong_as_double(old);
        if (cur >= val) break;
        assumed = old;
        old = atomicCAS(a, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

/* ── diagonal_term kernel (one thread per config, register accumulation) ── */

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
    (void)total_pairs;
    // Each diagonal element psi[i] is independent, so assign one thread per
    // config (grid-stride over configs) and accumulate the contribution of every
    // diagonal term in registers, writing psi[i] exactly once. This avoids the
    // L2 atomic contention of the naive (term, config) atomicAdd scheme without
    // needing a shared-memory reduction. `configs` here carry only the diagonal
    // term subset (flip_mask == 0), pre-filtered on the Python side.
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < B; i += stride) {
        const uint8_t* cfg = configs + i * n_qubytes;
        double acc_re = 0.0, acc_im = 0.0;
        for (int64_t t = 0; t < T; ++t) {
            if (!is_applicable<n_qubytes>(cfg, create_mask + t * n_qubytes,
                                          annihilate_mask + t * n_qubytes))
                continue;
            const uint8_t* fm = flip_mask + t * n_qubytes;
            bool is_diag = true;
            for (int q = 0; q < n_qubytes; ++q) { if (__ldg(fm + q)) { is_diag = false; break; } }
            if (!is_diag) continue;
            bool parity = jw_parity<n_qubytes>(cfg, parity_mask + t * n_qubytes, __ldg(parity_const + t));
            double sign = parity ? -1.0 : 1.0;
            acc_re += sign * __ldg(coef + t * 2);
            acc_im += sign * __ldg(coef + t * 2 + 1);
        }
        psi[i * 2]     = acc_re;
        psi[i * 2 + 1] = acc_im;
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
    // One thread per config; the kernel writes every psi[i], so no memset needed.
    int threads = 256, blocks = static_cast<int>(std::min<int64_t>((B + 255) / 256, 65535LL));
    if (blocks < 1) blocks = 1;
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
    int     occupied;
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
__global__ void apply_build_table_kernel(
    int64_t B_dst, const uint8_t* dst_configs,
    apply_hash_slot<n_qubytes>* table, int64_t cap)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < B_dst; i += stride) {
        const uint8_t* cfg = dst_configs + i * n_qubytes;
        uint64_t h = wyhash64<n_qubytes>(cfg, 0);
        int64_t slot_idx = h % cap;
        for (int64_t p = 0; p < cap; ++p) {
            apply_hash_slot<n_qubytes>& slot = table[slot_idx];
            if (atomicCAS(&slot.occupied, 0, 1) == 0) {
                for (int q = 0; q < n_qubytes; ++q) slot.key[q] = cfg[q];
                slot.index = i;
                break;
            }
            slot_idx = (slot_idx + 1) % cap;
        }
    }
}

// Applies H (direction 0) or H^dagger (direction 1) as a sparse matvec.
//
// The physical semantics are fixed by `direction` (which side is src/dst and
// whether coef is conjugated). Orthogonally, we may traverse EITHER side to
// minimise thread count (T * traversed_size): we always traverse the smaller
// of {src, dst} and build the linear-probe table over the other (looked-up)
// side. Because flip is self-inverse, both traversal modes enumerate exactly
// the same set of connected (src_config, dst_config) pairs, so psi_j is
// identical. `src_psi` is always indexed by the src-side position.
//
//   traverse_dst == 0: iterate src side (index m = src pos);
//       new_c = src XOR flip; out_idx = lookup(new_c in dst table); psi_idx = m
//   traverse_dst == 1: iterate dst side (index m = dst pos);
//       src_c = dst XOR flip; psi_idx = lookup(src_c in src table); out_idx = m
template <int n_qubytes>
__global__ void apply_within_kernel(
    int64_t B_trav, int64_t T, int64_t total,
    const uint8_t* traversed_configs, const double* src_psi,
    const uint8_t* create_mask, const uint8_t* annihilate_mask,
    const uint8_t* flip_mask, const uint8_t* parity_mask,
    const uint8_t* parity_const, const double* coef,
    const apply_hash_slot<n_qubytes>* lookup_table, int64_t lookup_cap,
    int direction, int traverse_dst, double* psi_j)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t k = idx; k < total; k += stride) {
        int64_t t = k / B_trav;
        int64_t m = k % B_trav;
        const uint8_t* tc = traversed_configs + m * n_qubytes;
        const uint8_t* fm = flip_mask + t * n_qubytes;

        // src config bytes (sc) and the two indices, depending on traversal side.
        uint8_t sc[n_qubytes];
        int64_t out_idx;
        int64_t psi_idx;
        if (traverse_dst == 0) {
            for (int q = 0; q < n_qubytes; ++q) sc[q] = tc[q];
            uint8_t new_c[n_qubytes];
            apply_flip<n_qubytes>(new_c, sc, fm);
            out_idx = apply_hash_lookup<n_qubytes>(new_c, lookup_table, lookup_cap);
            if (out_idx < 0) continue;
            psi_idx = m;
        } else {
            apply_flip<n_qubytes>(sc, tc, fm);  // sc = dst XOR flip
            psi_idx = apply_hash_lookup<n_qubytes>(sc, lookup_table, lookup_cap);
            if (psi_idx < 0) continue;
            out_idx = m;
        }

        // check config: forward uses sc, backward (H^dagger) uses sc XOR flip.
        uint8_t check_c[n_qubytes];
        if (direction == 1) {
            apply_flip<n_qubytes>(check_c, sc, fm);
        } else {
            for (int q = 0; q < n_qubytes; ++q) check_c[q] = sc[q];
        }
        if (!is_applicable<n_qubytes>(check_c, create_mask + t * n_qubytes, annihilate_mask + t * n_qubytes))
            continue;
        bool parity = jw_parity<n_qubytes>(check_c, parity_mask + t * n_qubytes, __ldg(parity_const + t));
        double sign = parity ? -1.0 : 1.0;
        double cr = __ldg(coef + t * 2);
        double ci = __ldg(coef + t * 2 + 1);
        if (direction == 1) ci = -ci;
        double pr = __ldg(src_psi + psi_idx * 2);
        double pi = __ldg(src_psi + psi_idx * 2 + 1);
        atomicAdd(psi_j + out_idx * 2,     sign * (cr * pr - ci * pi));
        atomicAdd(psi_j + out_idx * 2 + 1, sign * (cr * pi + ci * pr));
    }
}

ffi::Error ApplyWithinSubspaceImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8> configs_i, ffi::Buffer<ffi::F64> psi_i,
    ffi::Buffer<ffi::U8> configs_j,
    ffi::Buffer<ffi::U8> create_mask, ffi::Buffer<ffi::U8> annihilate_mask,
    ffi::Buffer<ffi::U8> flip_mask, ffi::Buffer<ffi::U8> parity_mask,
    ffi::Buffer<ffi::U8> parity_const, ffi::Buffer<ffi::F64> coef,
    int64_t direction, ffi::ResultBuffer<ffi::F64> psi_j)
{
    auto cd = configs_i.dimensions();
    int64_t B_i = cd[0], Q = cd[1];
    int64_t B_j = configs_j.dimensions()[0];
    int64_t T = create_mask.dimensions()[0];
    (void)Q;
    // direction 0: src = configs_i, dst = configs_j
    // direction 1: src = configs_j, dst = configs_i (H^dagger)
    const uint8_t* src_configs = (direction == 0) ? configs_i.typed_data() : configs_j.typed_data();
    const uint8_t* dst_configs = (direction == 0) ? configs_j.typed_data() : configs_i.typed_data();
    int64_t B_src = (direction == 0) ? B_i : B_j;
    int64_t B_dst = (direction == 0) ? B_j : B_i;
    const double* src_psi = reinterpret_cast<const double*>(psi_i.typed_data());

    // Traverse the smaller side (minimise T * traversed_size); build the
    // linear-probe table over the other (looked-up) side.
    int traverse_dst = (B_dst < B_src) ? 1 : 0;
    const uint8_t* traversed_configs = traverse_dst ? dst_configs : src_configs;
    const uint8_t* lookup_configs = traverse_dst ? src_configs : dst_configs;
    int64_t B_trav = traverse_dst ? B_dst : B_src;
    int64_t B_lookup = traverse_dst ? B_src : B_dst;

    int64_t hash_cap = static_cast<int64_t>(B_lookup / 0.6) + 1;
    apply_hash_slot<N_QUBYTES>* d_table = nullptr;
    cudaMallocAsync(&d_table, hash_cap * sizeof(apply_hash_slot<N_QUBYTES>), stream);
    cudaMemsetAsync(d_table, 0, hash_cap * sizeof(apply_hash_slot<N_QUBYTES>), stream);
    {
        int bt = 256, bb = static_cast<int>(std::min<int64_t>((B_lookup + 255) / 256, 65535LL));
        if (bb < 1) bb = 1;
        apply_build_table_kernel<N_QUBYTES><<<bb, bt, 0, stream>>>(B_lookup, lookup_configs, d_table, hash_cap);
    }
    cudaMemsetAsync(psi_j->untyped_data(), 0, psi_j->size_bytes(), stream);
    int64_t total = T * B_trav;
    int threads = 256, blocks = static_cast<int>(std::min<int64_t>((total + 255) / 256, 65535LL));
    if (blocks < 1) blocks = 1;
    apply_within_kernel<N_QUBYTES><<<blocks, threads, 0, stream>>>(
        B_trav, T, total,
        traversed_configs, src_psi,
        create_mask.typed_data(), annihilate_mask.typed_data(),
        flip_mask.typed_data(), parity_mask.typed_data(),
        parity_const.typed_data(), reinterpret_cast<const double*>(coef.typed_data()),
        d_table, hash_cap, direction, traverse_dst,
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
    .Arg<ffi::Buffer<ffi::F64>>().Attr<int64_t>("direction")
    .Ret<ffi::Buffer<ffi::F64>>());

/* ── find_all_relative_configs handler ── */

/* Exclude-set membership hash table (shared by find_all and find_topk).
   Replaces the O(exclude_size) linear scan per (term, config) pair with an
   O(1) linear-probe lookup. Built once per call over configs_exclude. */
template <int n_qubytes>
struct __align__(8) exclude_slot {
    uint8_t key[n_qubytes];
    int     occupied;
};

template <int n_qubytes>
__global__ void exclude_build_kernel(
    int64_t exclude_size, const uint8_t* exclude_configs,
    exclude_slot<n_qubytes>* table, int64_t cap)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t e = idx; e < exclude_size; e += stride) {
        const uint8_t* cfg = exclude_configs + e * n_qubytes;
        uint64_t h = wyhash64<n_qubytes>(cfg, 3);
        int64_t slot_idx = h % cap;
        for (int64_t p = 0; p < cap; ++p) {
            exclude_slot<n_qubytes>& slot = table[slot_idx];
            if (slot.occupied && [&] {
                    for (int q = 0; q < n_qubytes; ++q)
                        if (slot.key[q] != cfg[q]) return false;
                    return true;
                }()) {
                break;  // already present (duplicate exclude entry)
            }
            if (atomicCAS(&slot.occupied, 0, 1) == 0) {
                for (int q = 0; q < n_qubytes; ++q) slot.key[q] = cfg[q];
                break;
            }
            slot_idx = (slot_idx + 1) % cap;
        }
    }
}

template <int n_qubytes>
__device__ bool exclude_contains(
    const uint8_t* config, const exclude_slot<n_qubytes>* table, int64_t cap)
{
    if (cap == 0) return false;
    uint64_t h = wyhash64<n_qubytes>(config, 3);
    int64_t idx = h % cap;
    for (int64_t p = 0; p < cap; ++p) {
        const exclude_slot<n_qubytes>& slot = table[idx];
        if (!slot.occupied) return false;
        bool match = true;
        for (int q = 0; q < n_qubytes; ++q)
            if (slot.key[q] != config[q]) { match = false; break; }
        if (match) return true;
        idx = (idx + 1) % cap;
    }
    return false;
}

template <int n_qubytes>
struct __align__(8) findall_slot {
    uint8_t key[n_qubytes];
    double  real_val;
    double  imag_val;
    int     occupied;
    int     claimed;
};

template <int n_qubytes>
__global__ void find_all_kernel(
    int64_t B, int64_t T, int64_t total,
    const uint8_t* configs, const double* psi_i,
    const uint8_t* create_mask, const uint8_t* annihilate_mask,
    const uint8_t* flip_mask, const uint8_t* parity_mask,
    const uint8_t* parity_const, const double* coef,
    const exclude_slot<n_qubytes>* exclude_table, int64_t exclude_cap,
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
        // exclude check: O(1) hash-table membership
        if (exclude_contains<n_qubytes>(new_c, exclude_table, exclude_cap)) continue;
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
            if (atomicCAS(&slot.occupied, 0, 1) == 0) {
                // Claimed empty slot: publish key, then accumulate value.
                for (int q = 0; q < n_qubytes; ++q) slot.key[q] = new_c[q];
                __threadfence();
                slot.claimed = 1;
                atomicAdd(&slot.real_val, vr);
                atomicAdd(&slot.imag_val, vi);
                break;
            }
            // Occupied by another thread. If it is still mid-publish (claimed==0)
            // we cannot yet compare keys; skip to the next slot instead of
            // spin-waiting (a warp-divergent spin deadlocks under SIMT).
            if (slot.claimed) {
                bool match = true;
                for (int q = 0; q < n_qubytes; ++q)
                    if (slot.key[q] != new_c[q]) { match = false; break; }
                if (match) {
                    atomicAdd(&slot.real_val, vr);
                    atomicAdd(&slot.imag_val, vi);
                    break;
                }
            }
            slot_idx = (slot_idx + 1) % cap;
            if (p == 99) { *overflow = 1; }
        }
    }
}

/* Canonical slot for a key: the first slot (from wyhash(key) % cap, probing
   forward) that currently holds that key. All duplicate slots for the same key
   lie on the contiguous occupied run starting at that hash position (insert
   never skips an empty slot), so every duplicate resolves to the same canonical
   slot. Used to merge duplicates that concurrent insertion may have created
   (see the mid-publish skip in find_all_kernel). */
template <int n_qubytes>
__device__ int64_t find_canonical_slot(
    const findall_slot<n_qubytes>* table, int64_t cap, const uint8_t* key)
{
    uint64_t h = wyhash64<n_qubytes>(key, 1);
    int64_t idx = h % cap;
    for (int64_t p = 0; p < cap; ++p) {
        const findall_slot<n_qubytes>& slot = table[idx];
        if (!slot.occupied) return idx;  // unreachable for a key that exists
        bool match = true;
        for (int q = 0; q < n_qubytes; ++q)
            if (slot.key[q] != key[q]) { match = false; break; }
        if (match) return idx;
        idx = (idx + 1) % cap;
    }
    return idx;
}

/* Pass 1: each canonical slot claims one output index and writes its key. */
template <int n_qubytes>
__global__ void find_all_dedup_repr_kernel(
    const findall_slot<n_qubytes>* table, int64_t cap,
    uint8_t* new_configs, int* out_count, int64_t* slot_out)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t s = idx; s < cap; s += stride) {
        if (!table[s].occupied) continue;
        int64_t canonical = find_canonical_slot<n_qubytes>(table, cap, table[s].key);
        if (canonical == s) {
            int out = atomicAdd(out_count, 1);
            slot_out[s] = out;
            for (int q = 0; q < n_qubytes; ++q) new_configs[out * n_qubytes + q] = table[s].key[q];
        }
    }
}

/* Pass 2: every occupied slot adds its amplitude into its canonical's output. */
template <int n_qubytes>
__global__ void find_all_dedup_merge_kernel(
    const findall_slot<n_qubytes>* table, int64_t cap,
    double* psi_j, const int64_t* slot_out)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t s = idx; s < cap; s += stride) {
        if (!table[s].occupied) continue;
        int64_t canonical = find_canonical_slot<n_qubytes>(table, cap, table[s].key);
        int64_t out = slot_out[canonical];
        atomicAdd(psi_j + out * 2,     table[s].real_val);
        atomicAdd(psi_j + out * 2 + 1, table[s].imag_val);
    }
}

ffi::Error FindAllRelativeConfigsImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8> configs_i, ffi::Buffer<ffi::F64> psi_i,
    ffi::Buffer<ffi::U8> configs_exclude,
    ffi::Buffer<ffi::U8> create_mask, ffi::Buffer<ffi::U8> annihilate_mask,
    ffi::Buffer<ffi::U8> flip_mask, ffi::Buffer<ffi::U8> parity_mask,
    ffi::Buffer<ffi::U8> parity_const, ffi::Buffer<ffi::F64> coef,
    int64_t hash_capacity,
    ffi::ResultBuffer<ffi::U8> new_configs, ffi::ResultBuffer<ffi::F64> psi_j,
    ffi::ResultBuffer<ffi::S32> count, ffi::ResultBuffer<ffi::S32> overflow)
{
    auto cd = configs_i.dimensions();
    int64_t B = cd[0], Q = cd[1]; (void)Q;
    int64_t T = create_mask.dimensions()[0];
    int64_t cap = hash_capacity;
    int64_t total = T * B;
    int64_t exclude_size = configs_exclude.dimensions()[0];
    // allocate hash table
    findall_slot<N_QUBYTES>* d_table = nullptr;
    int* d_overflow = nullptr;
    int* d_count = nullptr;
    cudaMallocAsync(&d_table, cap * sizeof(findall_slot<N_QUBYTES>), stream);
    cudaMallocAsync(&d_overflow, sizeof(int), stream);
    cudaMallocAsync(&d_count, sizeof(int), stream);
    cudaMemsetAsync(d_table, 0, cap * sizeof(findall_slot<N_QUBYTES>), stream);
    cudaMemsetAsync(d_overflow, 0, sizeof(int), stream);
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    cudaMemsetAsync(new_configs->untyped_data(), 0, new_configs->size_bytes(), stream);
    cudaMemsetAsync(psi_j->untyped_data(), 0, psi_j->size_bytes(), stream);
    // Build exclude-set membership hash table (empty when exclude_size == 0).
    int64_t exclude_cap = exclude_size > 0 ? static_cast<int64_t>(exclude_size / 0.6) + 1 : 0;
    exclude_slot<N_QUBYTES>* d_exclude = nullptr;
    if (exclude_cap > 0) {
        cudaMallocAsync(&d_exclude, exclude_cap * sizeof(exclude_slot<N_QUBYTES>), stream);
        cudaMemsetAsync(d_exclude, 0, exclude_cap * sizeof(exclude_slot<N_QUBYTES>), stream);
        int eb = static_cast<int>(std::min<int64_t>((exclude_size + 255) / 256, 65535LL));
        if (eb < 1) eb = 1;
        exclude_build_kernel<N_QUBYTES><<<eb, 256, 0, stream>>>(
            exclude_size, configs_exclude.typed_data(), d_exclude, exclude_cap);
    }
    int threads = 256, blocks = static_cast<int>(std::min<int64_t>((total + 255) / 256, 65535LL));
    if (blocks < 1) blocks = 1;
    find_all_kernel<N_QUBYTES><<<blocks, threads, 0, stream>>>(
        B, T, total,
        configs_i.typed_data(), reinterpret_cast<const double*>(psi_i.typed_data()),
        create_mask.typed_data(), annihilate_mask.typed_data(),
        flip_mask.typed_data(), parity_mask.typed_data(),
        parity_const.typed_data(), reinterpret_cast<const double*>(coef.typed_data()),
        d_exclude, exclude_cap,
        d_table, cap, d_overflow);
    // Compact distinct configs into the output buffers. Concurrent insertion may
    // have placed the same key in multiple slots (mid-publish skip); dedup here
    // by merging every slot into its canonical slot's single output entry.
    int64_t* d_slot_out = nullptr;
    cudaMallocAsync(&d_slot_out, cap * sizeof(int64_t), stream);
    int cblocks = static_cast<int>(std::min<int64_t>((cap + 255) / 256, 65535LL));
    if (cblocks < 1) cblocks = 1;
    find_all_dedup_repr_kernel<N_QUBYTES><<<cblocks, 256, 0, stream>>>(
        d_table, cap,
        reinterpret_cast<uint8_t*>(new_configs->untyped_data()),
        d_count, d_slot_out);
    find_all_dedup_merge_kernel<N_QUBYTES><<<cblocks, 256, 0, stream>>>(
        d_table, cap,
        reinterpret_cast<double*>(psi_j->untyped_data()),
        d_slot_out);
    cudaMemcpyAsync(count->untyped_data(), d_count, sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(overflow->untyped_data(), d_overflow, sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaFreeAsync(d_table, stream);
    cudaFreeAsync(d_overflow, stream);
    cudaFreeAsync(d_count, stream);
    cudaFreeAsync(d_slot_out, stream);
    if (d_exclude) cudaFreeAsync(d_exclude, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FindAllRelativeConfigs, FindAllRelativeConfigsImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::F64>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::F64>>().Attr<int64_t>("hash_capacity")
    .Ret<ffi::Buffer<ffi::U8>>().Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::S32>>()
    .Ret<ffi::Buffer<ffi::S32>>());

/* ── find_topk_relative_configs handler ── */

template <int n_qubytes>
struct __align__(8) topk_slot {
    uint8_t key[n_qubytes];
    double  weight;
    int     occupied;
    int     claimed;
};

template <int n_qubytes>
__global__ void find_topk_kernel(
    int64_t B, int64_t T_chunk, int64_t term_start, int64_t total,
    const uint8_t* configs, const double* psi_i,
    const uint8_t* create_mask, const uint8_t* annihilate_mask,
    const uint8_t* flip_mask, const uint8_t* parity_mask,
    const uint8_t* parity_const, const double* coef,
    const exclude_slot<n_qubytes>* exclude_table, int64_t exclude_cap,
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
        // exclude check: O(1) hash-table membership
        if (exclude_contains<n_qubytes>(new_c, exclude_table, exclude_cap)) continue;
        uint64_t h = wyhash64<n_qubytes>(new_c, 2);
        int64_t slot_idx = h % cap;
        for (int64_t p = 0; p < cap && p < 100; ++p) {
            auto& slot = table[slot_idx];
            if (atomicCAS(&slot.occupied, 0, 1) == 0) {
                // We claimed an empty slot: publish key then weight.
                for (int q = 0; q < n_qubytes; ++q) slot.key[q] = new_c[q];
                __threadfence();
                atomic_max_double(&slot.weight, weight);
                break;
            }
            // Slot already claimed by someone. It may still be mid-publish; the
            // key compare below can spuriously fail, in which case we probe on.
            // A duplicate insertion at worst wastes a slot (cap = 2*K covers it)
            // and does not change the top-K weights after collection.
            bool match = true;
            for (int q = 0; q < n_qubytes; ++q)
                if (slot.key[q] != new_c[q]) { match = false; break; }
            if (match) {
                atomic_max_double(&slot.weight, weight);
                break;
            }
            slot_idx = (slot_idx + 1) % cap;
        }
    }
}

template <int n_qubytes>
__global__ void find_topk_collect_kernel(
    const topk_slot<n_qubytes>* table, int64_t cap,
    uint8_t* out_keys, double* out_weights)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t s = idx; s < cap; s += stride) {
        const topk_slot<n_qubytes>& slot = table[s];
        // Unoccupied slots keep weight 0 and zero key; they sort to the bottom.
        out_weights[s] = slot.occupied ? slot.weight : 0.0;
        for (int q = 0; q < n_qubytes; ++q)
            out_keys[s * n_qubytes + q] = slot.occupied ? slot.key[q] : 0;
    }
}

ffi::Error FindTopKRelativeConfigsImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8> configs_i, ffi::Buffer<ffi::F64> psi_i,
    int64_t count_selected, ffi::Buffer<ffi::U8> configs_exclude,
    ffi::Buffer<ffi::U8> create_mask, ffi::Buffer<ffi::U8> annihilate_mask,
    ffi::Buffer<ffi::U8> flip_mask, ffi::Buffer<ffi::U8> parity_mask,
    ffi::Buffer<ffi::U8> parity_const, ffi::Buffer<ffi::F64> coef,
    ffi::ResultBuffer<ffi::U8> table_keys, ffi::ResultBuffer<ffi::F64> table_weights)
{
    auto cd = configs_i.dimensions();
    int64_t B = cd[0], Q = cd[1]; (void)Q;
    int64_t T = create_mask.dimensions()[0];
    int64_t K = count_selected;
    int64_t cap = K * 2;
    int64_t total = T * B;
    int64_t exclude_size = configs_exclude.dimensions()[0];
    cudaMemsetAsync(table_keys->untyped_data(), 0, table_keys->size_bytes(), stream);
    cudaMemsetAsync(table_weights->untyped_data(), 0, table_weights->size_bytes(), stream);
    // allocate table + weight
    topk_slot<N_QUBYTES>* d_table = nullptr;
    double* d_min = nullptr;
    cudaMallocAsync(&d_table, cap * sizeof(topk_slot<N_QUBYTES>), stream);
    cudaMallocAsync(&d_min, sizeof(double), stream);
    cudaMemsetAsync(d_table, 0, cap * sizeof(topk_slot<N_QUBYTES>), stream);
    double zero = 0.0;
    cudaMemcpyAsync(d_min, &zero, sizeof(double), cudaMemcpyHostToDevice, stream);
    // Build exclude-set membership hash table (empty when exclude_size == 0).
    int64_t exclude_cap = exclude_size > 0 ? static_cast<int64_t>(exclude_size / 0.6) + 1 : 0;
    exclude_slot<N_QUBYTES>* d_exclude = nullptr;
    if (exclude_cap > 0) {
        cudaMallocAsync(&d_exclude, exclude_cap * sizeof(exclude_slot<N_QUBYTES>), stream);
        cudaMemsetAsync(d_exclude, 0, exclude_cap * sizeof(exclude_slot<N_QUBYTES>), stream);
        int eb = static_cast<int>(std::min<int64_t>((exclude_size + 255) / 256, 65535LL));
        if (eb < 1) eb = 1;
        exclude_build_kernel<N_QUBYTES><<<eb, 256, 0, stream>>>(
            exclude_size, configs_exclude.typed_data(), d_exclude, exclude_cap);
    }
    int threads = 256, blocks = static_cast<int>(std::min<int64_t>((total + 255) / 256, 65535LL));
    if (blocks < 1) blocks = 1;
    find_topk_kernel<N_QUBYTES><<<blocks, threads, 0, stream>>>(
        B, T, 0, total,
        configs_i.typed_data(), reinterpret_cast<const double*>(psi_i.typed_data()),
        create_mask.typed_data(), annihilate_mask.typed_data(),
        flip_mask.typed_data(), parity_mask.typed_data(),
        parity_const.typed_data(), reinterpret_cast<const double*>(coef.typed_data()),
        d_exclude, exclude_cap,
        d_table, cap, d_min);
    int cblocks = static_cast<int>(std::min<int64_t>((cap + 255) / 256, 65535LL));
    if (cblocks < 1) cblocks = 1;
    find_topk_collect_kernel<N_QUBYTES><<<cblocks, 256, 0, stream>>>(
        d_table, cap,
        reinterpret_cast<uint8_t*>(table_keys->untyped_data()),
        reinterpret_cast<double*>(table_weights->untyped_data()));
    cudaFreeAsync(d_table, stream);
    cudaFreeAsync(d_min, stream);
    if (d_exclude) cudaFreeAsync(d_exclude, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FindTopKRelativeConfigs, FindTopKRelativeConfigsImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::F64>>()
    .Attr<int64_t>("count_selected").Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::F64>>()
    .Ret<ffi::Buffer<ffi::U8>>().Ret<ffi::Buffer<ffi::F64>>());
