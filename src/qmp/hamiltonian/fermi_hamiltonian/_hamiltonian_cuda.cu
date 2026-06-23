/* Fermi Hamiltonian CUDA kernel.

Template parameter N_QUBYTES = ceil(n_qubits/8).
Compiled per-n_qubytes by nvcc -DN_QUBYTES=X.
XLA FFI handlers for four operations.
*/

#include <cstdint>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

/* ── Device utilities ── */

template <int N_QUBYTES>
__device__ bool is_applicable(
    const uint8_t* config, const uint8_t* cm, const uint8_t* am)
{
    for (int q = 0; q < N_QUBYTES; ++q) {
        if ((config[q] & cm[q]) != 0) return false;
        if ((config[q] & am[q]) != am[q]) return false;
    }
    return true;
}

template <int N_QUBYTES>
__device__ void apply_flip(uint8_t* dst, const uint8_t* src, const uint8_t* fm)
{
    for (int q = 0; q < N_QUBYTES; ++q) dst[q] = src[q] ^ fm[q];
}

template <int N_QUBYTES>
__device__ bool jw_parity(
    const uint8_t* config, const uint8_t* pm, uint8_t pc)
{
    uint8_t p = pc;
    for (int q = 0; q < N_QUBYTES; ++q)
        p ^= __popc(static_cast<unsigned>(pm[q] & config[q])) & 1;
    return p & 1;
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

/* ── diagonal_term kernel ── */

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
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t k = idx; k < total_pairs; k += stride) {
        int64_t t = k / B;
        int64_t i = k % B;
        const uint8_t* cfg = configs + i * N_QUBYTES;
        if (!is_applicable<N_QUBYTES>(cfg, create_mask + t * N_QUBYTES,
                                       annihilate_mask + t * N_QUBYTES))
            continue;
        const uint8_t* fm = flip_mask + t * N_QUBYTES;
        bool is_diag = true;
        for (int q = 0; q < N_QUBYTES; ++q) { if (fm[q]) { is_diag = false; break; } }
        if (!is_diag) continue;
        bool parity = jw_parity<N_QUBYTES>(cfg, parity_mask + t * N_QUBYTES, parity_const[t]);
        double sign = parity ? -1.0 : 1.0;
        atomicAdd(psi + i * 2,     sign * coef[t * 2]);
        atomicAdd(psi + i * 2 + 1, sign * coef[t * 2 + 1]);
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
    cudaMemsetAsync(psi->untyped_data(), 0, psi->size_bytes(), stream);
    int threads = 256, blocks = static_cast<int>(std::min<int64_t>((total + 255) / 256, 65535LL));
    diagonal_term_kernel<N_QUBYTES><<<blocks, threads, 0, stream>>>(
        B, T, total, configs.typed_data(), create_mask.typed_data(),
        annihilate_mask.typed_data(), flip_mask.typed_data(),
        parity_mask.typed_data(), parity_const.typed_data(),
        reinterpret_cast<const double*>(coef.typed_data()),
        reinterpret_cast<double*>(psi->typed_data()));
    (void)Q;
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

/* ── apply_within, find_all, find_topk stubs ──
   Full implementations deferred per plan tasks 8b-8d.
   Supplement 6 requires if constexpr (N_QUBYTES <= 8) optimization.
   Supplements 1-3 require block reduction, __ldg(), hash table caching.
*/

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
