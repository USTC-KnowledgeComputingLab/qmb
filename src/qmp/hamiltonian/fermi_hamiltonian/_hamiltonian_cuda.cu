/* Fermi Hamiltonian CUDA kernel.

Template parameter N_QUBYTES = ceil(n_qubits/8).
Compiled per-n_qubytes by nvcc -DN_QUBYTES=X.
XLA FFI handlers for four operations.

Optimizations:
- if constexpr (N_QUBYTES <= 8): uint64_t register path
- __ldg() read-only cache for all input data
- Block-level shared-memory reduction for diagonal_term
- wyhash64 inline hash function
*/

#include <cstdint>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

/* ── Device utilities with static dispatch ── */

template <int N_QUBYTES>
__device__ bool is_applicable(
    const uint8_t* config, const uint8_t* cm, const uint8_t* am)
{
    if constexpr (N_QUBYTES <= 8) {
        const uint64_t* c64 = reinterpret_cast<const uint64_t*>(config);
        const uint64_t* m64 = reinterpret_cast<const uint64_t*>(cm);
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(am);
        if ((*(c64) & *(m64)) != 0) return false;
        if ((*(c64) & *(a64)) != *(a64)) return false;
        for (int q = 8; q < N_QUBYTES; ++q) {
            if ((config[q] & cm[q]) != 0) return false;
            if ((config[q] & am[q]) != am[q]) return false;
        }
        return true;
    } else {
        for (int q = 0; q < N_QUBYTES; ++q) {
            if ((__ldg(config + q) & __ldg(cm + q)) != 0) return false;
            if ((__ldg(config + q) & __ldg(am + q)) != __ldg(am + q)) return false;
        }
        return true;
    }
}

template <int N_QUBYTES>
__device__ void apply_flip(uint8_t* dst, const uint8_t* src, const uint8_t* fm)
{
    if constexpr (N_QUBYTES <= 8) {
        uint64_t* d64 = reinterpret_cast<uint64_t*>(dst);
        const uint64_t* s64 = reinterpret_cast<const uint64_t*>(src);
        const uint64_t* f64 = reinterpret_cast<const uint64_t*>(fm);
        *d64 = *s64 ^ *f64;
        for (int q = 8; q < N_QUBYTES; ++q)
            dst[q] = src[q] ^ __ldg(fm + q);
    } else {
        for (int q = 0; q < N_QUBYTES; ++q)
            dst[q] = __ldg(src + q) ^ __ldg(fm + q);
    }
}

template <int N_QUBYTES>
__device__ bool jw_parity(
    const uint8_t* config, const uint8_t* pm, uint8_t pc)
{
    if constexpr (N_QUBYTES <= 8) {
        const uint64_t* c64 = reinterpret_cast<const uint64_t*>(config);
        const uint64_t* p64 = reinterpret_cast<const uint64_t*>(pm);
        uint8_t p = pc ^ (__popcll(*(c64) & *(p64)) & 1);
        for (int q = 8; q < N_QUBYTES; ++q)
            p ^= __popc(static_cast<unsigned>(__ldg(pm + q) & __ldg(config + q))) & 1;
        return p & 1;
    } else {
        uint8_t p = pc;
        for (int q = 0; q < N_QUBYTES; ++q)
            p ^= __popc(static_cast<unsigned>(__ldg(pm + q) & __ldg(config + q))) & 1;
        return p & 1;
    }
}

/* ── wyhash64 inline ── */

template <int N_QUBYTES>
__device__ uint64_t wyhash64(const uint8_t* key, uint64_t seed) {
    uint64_t a = seed ^ N_QUBYTES;
    for (int q = 0; q < static_cast<int>(N_QUBYTES / 8); ++q) {
        uint64_t v;
        __builtin_memcpy(&v, key + q * 8, 8);
        a ^= v; a *= 0x9ddfea08eb382d59ULL; a ^= (a >> 28);
    }
    int rem = N_QUBYTES % 8;
    if (rem) {
        uint64_t v = 0;
        __builtin_memcpy(&v, key + (N_QUBYTES - rem), rem);
        a ^= v; a *= 0x9ddfea08eb382d59ULL; a ^= (a >> 28);
    }
    a *= 0x9ddfea08eb382d59ULL; a ^= (a >> 28);
    return a;
}

/* ── diagonal_term kernel with block-level reduction ── */

template <int N_QUBYTES>
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
    __shared__ double s_re[256];
    __shared__ double s_im[256];
    const int tid = threadIdx.x;
    s_re[tid] = 0.0;
    s_im[tid] = 0.0;
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    // Accumulate contributions per config to shared memory
    for (int64_t k = idx; k < total_pairs; k += stride) {
        int64_t t = k / B;
        int64_t i = k % B;
        const uint8_t* cfg = configs + i * N_QUBYTES;
        if (!is_applicable<N_QUBYTES>(cfg, create_mask + t * N_QUBYTES,
                                       annihilate_mask + t * N_QUBYTES))
            continue;
        const uint8_t* fm = flip_mask + t * N_QUBYTES;
        bool is_diag = true;
        for (int q = 0; q < N_QUBYTES; ++q) { if (__ldg(fm + q)) { is_diag = false; break; } }
        if (!is_diag) continue;
        bool parity = jw_parity<N_QUBYTES>(cfg, parity_mask + t * N_QUBYTES, __ldg(parity_const + t));
        double sign = parity ? -1.0 : 1.0;
        // accumulate to per-config shared memory using atomicAdd
        atomicAdd(&s_re[i % 256], sign * __ldg(coef + t * 2));
        atomicAdd(&s_im[i % 256], sign * __ldg(coef + t * 2 + 1));
    }
    __syncthreads();

    // write block-accumulated results to global
    for (int i = tid; i < B; i += blockDim.x) {
        if (s_re[i % 256] != 0.0 || s_im[i % 256] != 0.0) {
            atomicAdd(psi + i * 2,     s_re[i % 256]);
            atomicAdd(psi + i * 2 + 1, s_im[i % 256]);
        }
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

/* ── apply_within, find_all, find_topk stubs ── */

ffi::Error ApplyWithinSubspaceImpl(
    cudaStream_t, ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::F64>,
    ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>,
    ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>,
    ffi::Buffer<ffi::F64>, ffi::Attr<int32_t>,
    ffi::ResultBuffer<ffi::F64>)
{ return ffi::Error::Unimplemented("apply_within not yet implemented"); }

XLA_FFI_DEFINE_HANDLER_SYMBOL(ApplyWithinSubspace, ApplyWithinSubspaceImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::F64>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::F64>>().Attr<int32_t>("direction")
    .Ret<ffi::Buffer<ffi::F64>>());

ffi::Error FindAllRelativeConfigsImpl(
    cudaStream_t, ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::F64>,
    ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>,
    ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>,
    ffi::Buffer<ffi::F64>, ffi::Attr<int32_t>,
    ffi::ResultBuffer<ffi::U8>, ffi::ResultBuffer<ffi::F64>, ffi::ResultBuffer<ffi::S32>)
{ return ffi::Error::Unimplemented("find_all not yet implemented"); }

XLA_FFI_DEFINE_HANDLER_SYMBOL(FindAllRelativeConfigs, FindAllRelativeConfigsImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::F64>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::F64>>().Attr<int32_t>("hash_capacity")
    .Ret<ffi::Buffer<ffi::U8>>().Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::S32>>());

ffi::Error FindTopKRelativeConfigsImpl(
    cudaStream_t, ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::F64>,
    ffi::Attr<int32_t>, ffi::Buffer<ffi::U8>,
    ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>,
    ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::F64>,
    ffi::ResultBuffer<ffi::U8>)
{ return ffi::Error::Unimplemented("find_topk not yet implemented"); }

XLA_FFI_DEFINE_HANDLER_SYMBOL(FindTopKRelativeConfigs, FindTopKRelativeConfigsImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::F64>>()
    .Attr<int32_t>("count_selected").Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::F64>>()
    .Ret<ffi::Buffer<ffi::U8>>());
