#include <cstdint>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// =============================================================================
// 工具函数
// =============================================================================

__device__ inline std::uint8_t popcount_parity(std::uint8_t byte) {
    return __popc(static_cast<unsigned int>(byte)) & 1;
}

__device__ inline bool get_bit(const std::uint8_t* data, std::int64_t index) {
    return (data[index >> 3] >> (index & 7)) & 1;
}

__device__ inline void set_bit(std::uint8_t* data, std::int64_t index, bool value) {
    if (value)
        data[index >> 3] |= (1 << (index & 7));
    else
        data[index >> 3] &= ~(1 << (index & 7));
}

/// 实现: applicable = (config & create_mask) == 0 && (config & annihilate_mask) == annihilate_mask
__device__ inline bool is_applicable(
    const std::uint8_t* config,
    const std::uint8_t* create_mask,
    const std::uint8_t* annihilate_mask,
    std::int64_t n_qubytes)
{
    for (std::int64_t q = 0; q < n_qubytes; ++q) {
        if ((config[q] & create_mask[q]) != 0) return false;
        if ((config[q] & annihilate_mask[q]) != annihilate_mask[q]) return false;
    }
    return true;
}

/// 实现: new_config = config ^ flip_mask
__device__ inline void apply_flip(
    std::uint8_t* config,
    const std::uint8_t* flip_mask,
    std::int64_t n_qubytes)
{
    for (std::int64_t q = 0; q < n_qubytes; ++q) {
        config[q] ^= flip_mask[q];
    }
}

/// 计算 JW 奇偶性: parity_const ^ (popcount(parity_mask & config) & 1)
/// 返回 true 表示负号, false 表示正号。
__device__ inline bool jw_parity(
    const std::uint8_t* config,
    const std::uint8_t* parity_mask,
    std::uint8_t parity_const,
    std::int64_t n_qubytes)
{
    std::uint8_t p = parity_const;
    for (std::int64_t q = 0; q < n_qubytes; ++q) {
        p ^= popcount_parity(parity_mask[q] & config[q]);
    }
    return p & 1;
}

// =============================================================================
// diagonal_term — 对角元
// =============================================================================

template <std::int64_t n_qubytes>
__global__ void diagonal_term_kernel(
    std::int64_t batch_size,
    std::int64_t term_number,
    const std::uint8_t* __restrict__ configs,         // [B, Q]
    const std::uint8_t* __restrict__ create_mask,      // [T, Q]
    const std::uint8_t* __restrict__ annihilate_mask,  // [T, Q]
    const std::uint8_t* __restrict__ flip_mask,        // [T, Q]
    const std::uint8_t* __restrict__ parity_mask,      // [T, Q]
    const std::uint8_t* __restrict__ parity_const,     // [T]
    const double*    __restrict__ coef,                // [T, 2]
    double*          __restrict__ psi)                 // [B, 2]
{
    std::int64_t term_idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::int64_t batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (term_idx >= term_number || batch_idx >= batch_size) return;

    const std::uint8_t* config = configs + batch_idx * n_qubytes;
    const std::uint8_t* cm = create_mask + term_idx * n_qubytes;
    const std::uint8_t* am = annihilate_mask + term_idx * n_qubytes;
    const std::uint8_t* fm = flip_mask + term_idx * n_qubytes;

    // 可作用性
    if (!is_applicable(config, cm, am, n_qubytes)) return;

    // 对角条件: 无翻转
    for (std::int64_t q = 0; q < n_qubytes; ++q) {
        if (fm[q] != 0) return;
    }

    bool parity = jw_parity(config, parity_mask + term_idx * n_qubytes,
                            parity_const[term_idx], n_qubytes);
    double sign = parity ? -1.0 : 1.0;
    double cr = coef[term_idx * 2];
    double ci = coef[term_idx * 2 + 1];

    atomicAdd(&psi[batch_idx * 2],     sign * cr);
    atomicAdd(&psi[batch_idx * 2 + 1], sign * ci);
}

ffi::Error DiagonalTermImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8>  configs,
    ffi::Buffer<ffi::U8>  create_mask,
    ffi::Buffer<ffi::U8>  annihilate_mask,
    ffi::Buffer<ffi::U8>  flip_mask,
    ffi::Buffer<ffi::U8>  parity_mask,
    ffi::Buffer<ffi::U8>  parity_const,
    ffi::Buffer<ffi::F64> coef,
    ffi::ResultBuffer<ffi::F64> psi)
{
    auto cd = configs.dimensions();
    std::int64_t B = cd[0];
    std::int64_t Q = cd[1];
    auto td = create_mask.dimensions();
    std::int64_t T = td[0];

    cudaMemsetAsync(psi->untyped_data(), 0, psi->size_bytes(), stream);

    constexpr std::int64_t nq = 0;  // 动态 n_qubytes — 使用 switch 分发
    dim3 threads(1, 512);
    dim3 blocks((T + threads.x - 1) / threads.x, (B + threads.y - 1) / threads.y);
    diagonal_term_kernel<0><<<blocks, threads, 0, stream>>>(
        B, T,
        configs.typed_data(), create_mask.typed_data(),
        annihilate_mask.typed_data(), flip_mask.typed_data(),
        parity_mask.typed_data(), parity_const.typed_data(),
        reinterpret_cast<const double*>(coef.typed_data()),
        reinterpret_cast<double*>(psi->typed_data()));

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DiagonalTerm, DiagonalTermImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);

// =============================================================================
// apply_within — 稀疏 H·psi 投影
// =============================================================================

ffi::Error ApplyWithinImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8>  configs_i,
    ffi::Buffer<ffi::F64> psi_i,
    ffi::Buffer<ffi::U8>  configs_j,
    ffi::Buffer<ffi::U8>  create_mask,
    ffi::Buffer<ffi::U8>  annihilate_mask,
    ffi::Buffer<ffi::U8>  flip_mask,
    ffi::Buffer<ffi::U8>  parity_mask,
    ffi::Buffer<ffi::U8>  parity_const,
    ffi::Buffer<ffi::F64> coef,
    ffi::Attr<std::int32_t> direction,
    ffi::ResultBuffer<ffi::F64> psi_j)
{
    // TODO: 实现 forward/backward apply_within kernel
    // 见 NOTE.md 4.3 节 — 需要:
    // 1. 排序 configs_j (forward) 或 configs_i (backward)
    // 2. 二级索引二分查找
    // 3. 2D grid kernel + atomicAdd
    cudaMemsetAsync(psi_j->untyped_data(), 0, psi_j->size_bytes(), stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ApplyWithin, ApplyWithinImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Attr<std::int32_t>("direction")
        .Ret<ffi::Buffer<ffi::F64>>()
);

// =============================================================================
// list_relative — 全部列举 + 哈希表去重
// =============================================================================

ffi::Error ListRelativeImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8>  configs_i,
    ffi::Buffer<ffi::F64> psi_i,
    ffi::Buffer<ffi::U8>  configs_exclude,
    ffi::Buffer<ffi::U8>  create_mask,
    ffi::Buffer<ffi::U8>  annihilate_mask,
    ffi::Buffer<ffi::U8>  flip_mask,
    ffi::Buffer<ffi::U8>  parity_mask,
    ffi::Buffer<ffi::U8>  parity_const,
    ffi::Buffer<ffi::F64> coef,
    ffi::Attr<std::int32_t> hash_capacity,
    ffi::ResultBuffer<ffi::U8>  new_configs,
    ffi::ResultBuffer<ffi::F64> psi_j,
    ffi::ResultBuffer<ffi::S32> count)
{
    // TODO: 实现 list_relative kernel
    // 见 NOTE.md 4.4 节 — 需要:
    // 1. cuCollections cuco::static_map (config_bytes -> (real, imag))
    // 2. Bloom filter 预筛查 exclude_configs
    // 3. 哈希表插入 (atomicAdd for matching configs, CAS for new configs)
    // 4. 线性扫描收集结果
    // 5. 设置 count 为实际输出数量
    cudaMemsetAsync(new_configs->untyped_data(), 0, new_configs->size_bytes(), stream);
    cudaMemsetAsync(psi_j->untyped_data(), 0, psi_j->size_bytes(), stream);
    *reinterpret_cast<std::int32_t*>(count->untyped_data()) = 0;
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ListRelative, ListRelativeImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Attr<std::int32_t>("hash_capacity")
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::S32>>()
);

// =============================================================================
// find_relative — 流式 Top-K 选择
// =============================================================================

ffi::Error FindRelativeImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8>  configs_i,
    ffi::Buffer<ffi::F64> psi_i,
    ffi::Attr<std::int32_t> count_selected,
    ffi::Buffer<ffi::U8>  configs_exclude,
    ffi::Buffer<ffi::U8>  create_mask,
    ffi::Buffer<ffi::U8>  annihilate_mask,
    ffi::Buffer<ffi::U8>  flip_mask,
    ffi::Buffer<ffi::U8>  parity_mask,
    ffi::Buffer<ffi::U8>  parity_const,
    ffi::Buffer<ffi::F64> coef,
    ffi::ResultBuffer<ffi::U8>  new_configs)
{
    // TODO: 实现 find_relative kernel
    // 见 NOTE.md 4.5 节 — 需要:
    // 1. 全局哈希表 (容量 2K, L2 cached) + 阈值加速
    // 2. 流式插入: 阈值检查 (快速拒绝) → 哈希表查找/插入
    // 3. 周期性 compact: CUB radix sort → 取 top K → 重建哈希表
    // 4. 输出 top K unique configs
    cudaMemsetAsync(new_configs->untyped_data(), 0, new_configs->size_bytes(), stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FindRelative, FindRelativeImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Attr<std::int32_t>("count_selected")
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::U8>>()
);
