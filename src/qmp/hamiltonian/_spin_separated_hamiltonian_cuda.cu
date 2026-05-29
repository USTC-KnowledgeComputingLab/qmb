#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <utility>

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <torch/extension.h>

namespace {

__device__ inline std::uint8_t popcount_parity(std::uint8_t byte) {
    return __popc(static_cast<unsigned int>(byte)) & 1;
}

template <std::int64_t size>
struct array_less {
    __host__ __device__ bool operator()(
        const std::array<std::uint8_t, size>& lhs,
        const std::array<std::uint8_t, size>& rhs) const {
        for (std::int64_t i = 0; i < size; ++i) {
            if (lhs[i] < rhs[i]) return true;
            if (lhs[i] > rhs[i]) return false;
        }
        return false;
    }
};

__device__ inline bool get_bit(const std::uint8_t* data, std::int64_t index) {
    return (data[index >> 3] >> (index & 7)) & 1;
}

__device__ inline void set_bit(std::uint8_t* data, std::int64_t index, bool value) {
    if (value)
        data[index >> 3] |= (1 << (index & 7));
    else
        data[index >> 3] &= ~(1 << (index & 7));
}

template <std::int64_t n_qubytes>
__device__ std::uint8_t jw_parity(const std::uint8_t* config, std::int64_t site) {
    std::uint8_t p = 0;
    std::int64_t bi = static_cast<std::int64_t>(site >> 3);
    for (std::int64_t b = 0; b < bi; ++b)
        p ^= popcount_parity(config[b]);
    std::uint8_t mask = static_cast<std::uint8_t>((1 << (site & 7)) - 1);
    p ^= popcount_parity(config[bi] & mask);
    return p;
}

template <std::int64_t n_qubytes, std::int64_t particle_cut, std::int64_t max_op_number, bool forward>
__device__ std::pair<bool, bool> hamiltonian_apply_kernel(
    std::array<std::uint8_t, n_qubytes>& current_configs,
    std::int64_t term_index,
    const std::array<std::int16_t, max_op_number>* site,
    const std::array<std::uint8_t, max_op_number>* kind)
{
    static_assert(particle_cut == 1 || particle_cut == 2);
    bool success = true;
    bool parity = false;
    if constexpr (forward) {
        for (std::int64_t i = max_op_number; i-- > 0;) {
            std::uint8_t k = kind[term_index][i];
            if (k == 2) continue;
            std::int16_t s = site[term_index][i];
            if (get_bit(current_configs.data(), s) == k) { success = false; break; }
            set_bit(current_configs.data(), s, k);
            if constexpr (particle_cut == 1)
                parity ^= jw_parity<n_qubytes>(current_configs.data(), s);
        }
    } else {
        for (std::int64_t i = 0; i < max_op_number; ++i) {
            std::uint8_t k = kind[term_index][i];
            if (k == 2) continue;
            std::int16_t s = site[term_index][i];
            bool target = 1 - k;
            if (get_bit(current_configs.data(), s) == target) { success = false; break; }
            set_bit(current_configs.data(), s, target);
            if constexpr (particle_cut == 1)
                parity ^= jw_parity<n_qubytes>(current_configs.data(), s);
        }
    }
    return std::make_pair(success, parity);
}

template <std::int64_t n_qubytes, std::int64_t particle_cut, std::int64_t max_op_number, bool forward>
__device__ void apply_within_subspace_in_double_side_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t dst_batch_size,
    const std::array<std::int16_t, max_op_number>* site,
    const std::array<std::uint8_t, max_op_number>* kind,
    const std::array<double, 2>* coef,
    const std::array<std::uint8_t, n_qubytes>* src_configs,
    const std::array<double, 2>* src_psi,
    const std::array<std::uint8_t, n_qubytes>* sorted_dst_configs,
    std::array<double, 2>* result_psi)
{
    std::array<std::uint8_t, n_qubytes> current_configs = src_configs[batch_index];
    auto [success, parity] = hamiltonian_apply_kernel<n_qubytes, particle_cut, max_op_number, forward>(
        current_configs, term_index, site, kind);
    if (!success) return;
    auto less = array_less<n_qubytes>();
    std::int64_t lo = 0, hi = dst_batch_size - 1;
    bool found = false;
    std::int64_t mid = 0;
    while (lo <= hi) {
        mid = (lo + hi) / 2;
        if (less(current_configs, sorted_dst_configs[mid])) hi = mid - 1;
        else if (less(sorted_dst_configs[mid], current_configs)) lo = mid + 1;
        else { found = true; break; }
    }
    if (!found) return;
    std::int8_t sign = parity ? -1 : +1;
    double r = sign * (coef[term_index][0] * src_psi[batch_index][0] - coef[term_index][1] * src_psi[batch_index][1]);
    double i = sign * (coef[term_index][0] * src_psi[batch_index][1] + coef[term_index][1] * src_psi[batch_index][0]);
    atomicAdd(&result_psi[mid][0], r);
    atomicAdd(&result_psi[mid][1], i);
}

template <std::int64_t n_qubytes, std::int64_t particle_cut, std::int64_t max_op_number, bool forward>
__global__ void apply_within_subspace_in_double_side_kernel_interface(
    std::int64_t term_number,
    std::int64_t src_batch_size,
    std::int64_t dst_batch_size,
    const std::array<std::int16_t, max_op_number>* site,
    const std::array<std::uint8_t, max_op_number>* kind,
    const std::array<double, 2>* coef,
    const std::array<std::uint8_t, n_qubytes>* src_configs,
    const std::array<double, 2>* src_psi,
    const std::array<std::uint8_t, n_qubytes>* sorted_dst_configs,
    std::array<double, 2>* result_psi)
{
    std::int64_t term_index = blockIdx.x * blockDim.x + threadIdx.x;
    std::int64_t batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (term_index >= term_number || batch_index >= src_batch_size) return;
    apply_within_subspace_in_double_side_kernel<n_qubytes, particle_cut, max_op_number, forward>(
        term_index, batch_index, dst_batch_size, site, kind, coef, src_configs, src_psi, sorted_dst_configs, result_psi);
}

__global__ void separate_interleaved_to_block_kernel(
    const std::uint8_t* interleaved, std::int64_t batch_size, std::int64_t n_qubytes,
    std::uint8_t* block, std::int64_t n_up)
{
    std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const std::uint8_t* src = interleaved + idx * n_qubytes;
    std::uint8_t* dst = block + idx * n_qubytes;
    std::int64_t n_total = n_qubytes * 8;
    for (std::int64_t i = 0; i < n_total; ++i) {
        std::int64_t block_site = (i % 2 == 0) ? i / 2 : n_up + i / 2;
        if (get_bit(src, i)) set_bit(dst, block_site, true);
    }
}

template <std::int64_t n_qubytes, std::int64_t particle_cut, std::int64_t max_op_number>
auto sort_configs_cuda(const torch::Tensor& configs, cudaStream_t stream) -> std::tuple<torch::Tensor, torch::Tensor> {
    using Config = std::array<std::uint8_t, n_qubytes>;
    std::int64_t n = configs.size(0);
    auto sort_idx = torch::arange(n, torch::TensorOptions().dtype(torch::kInt64).device(configs.device()));
    auto sorted = configs.clone();
    auto* sorted_ptr = reinterpret_cast<Config*>(sorted.data_ptr());
    auto* idx_ptr = sort_idx.data_ptr<std::int64_t>();
    thrust::sort_by_key(thrust::device.on(stream), sorted_ptr, sorted_ptr + n, idx_ptr, array_less<n_qubytes>());
    return {sorted, sort_idx};
}

template <std::int64_t n_qubytes, std::int64_t particle_cut, std::int64_t max_op_number, std::int64_t n_up>
auto apply_within_subspace_in_double_side_interface(
    const torch::Tensor& configs_i,
    const torch::Tensor& psi_i,
    const torch::Tensor& configs_j,
    const torch::Tensor& site,
    const torch::Tensor& kind,
    const torch::Tensor& coef,
    bool configs_i_sorted,
    bool configs_j_sorted,
    int64_t direction) -> torch::Tensor
{
    using Config = std::array<std::uint8_t, n_qubytes>;
    using Coef2 = std::array<double, 2>;

    std::int64_t batch_i = configs_i.size(0);
    std::int64_t batch_j = configs_j.size(0);
    std::int64_t term_number = site.size(0);

    std::int64_t device_id = configs_i.device().index();
    at::cuda::CUDAGuard cuda_device_guard(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);

    // Separate interleaved configs to block order (GPU kernel)
    auto block_i = torch::empty({batch_i, n_qubytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(configs_i.device()));
    auto block_j = torch::empty({batch_j, n_qubytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(configs_j.device()));
    constexpr int kSepBlockSize = 256;
    separate_interleaved_to_block_kernel<<<(batch_i + kSepBlockSize - 1) / kSepBlockSize, kSepBlockSize, 0, stream>>>(
        configs_i.data_ptr<std::uint8_t>(), batch_i, n_qubytes, block_i.data_ptr<std::uint8_t>(), n_up);
    separate_interleaved_to_block_kernel<<<(batch_j + kSepBlockSize - 1) / kSepBlockSize, kSepBlockSize, 0, stream>>>(
        configs_j.data_ptr<std::uint8_t>(), batch_j, n_qubytes, block_j.data_ptr<std::uint8_t>(), n_up);

    const auto* site_ptr = reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr());
    const auto* kind_ptr = reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr());
    const auto* coef_ptr = reinterpret_cast<const Coef2*>(coef.data_ptr());

    auto result_psi = torch::zeros({batch_j, 2},
        torch::TensorOptions().dtype(torch::kFloat64).device(configs_i.device()));
    auto* result_ptr = reinterpret_cast<Coef2*>(result_psi.data_ptr());

    cudaDeviceProp prop;
    AT_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::int64_t max_threads_per_block = prop.maxThreadsPerBlock;
    auto threads_per_block = dim3{1, static_cast<unsigned int>(max_threads_per_block >> 1)};

    if (direction == 0) {
        torch::Tensor sorted_j, sort_j_idx;
        if (configs_j_sorted) {
            sorted_j = block_j;
        } else {
            std::tie(sorted_j, sort_j_idx) = sort_configs_cuda<n_qubytes, particle_cut, max_op_number>(block_j, stream);
        }
        const auto* sorted_j_ptr = reinterpret_cast<const Config*>(sorted_j.data_ptr());
        const auto* bi_ptr = reinterpret_cast<const Config*>(block_i.data_ptr());
        const auto* pi_ptr = reinterpret_cast<const Coef2*>(psi_i.data_ptr());
        auto num_blocks = dim3{
            static_cast<unsigned int>((term_number + threads_per_block.x - 1) / threads_per_block.x),
            static_cast<unsigned int>((batch_i + threads_per_block.y - 1) / threads_per_block.y),
        };
        apply_within_subspace_in_double_side_kernel_interface<n_qubytes, particle_cut, max_op_number, true>
            <<<num_blocks, threads_per_block, 0, stream>>>(
                term_number, batch_i, batch_j, site_ptr, kind_ptr, coef_ptr, bi_ptr, pi_ptr, sorted_j_ptr, result_ptr);
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!configs_j_sorted) {
            auto unsorted = torch::zeros_like(result_psi);
            unsorted.index_put_({sort_j_idx}, result_psi);
            return unsorted;
        }
        return result_psi;
    } else {
        torch::Tensor sorted_i, sort_i_idx;
        if (configs_i_sorted) {
            sorted_i = block_i;
        } else {
            std::tie(sorted_i, sort_i_idx) = sort_configs_cuda<n_qubytes, particle_cut, max_op_number>(block_i, stream);
        }
        torch::Tensor sorted_psi_i;
        if (configs_i_sorted) {
            sorted_psi_i = psi_i;
        } else {
            sorted_psi_i = torch::zeros({batch_i, 2},
                torch::TensorOptions().dtype(torch::kFloat64).device(configs_i.device()));
            sorted_psi_i.index_put_({sort_i_idx}, psi_i);
        }
        const auto* sorted_i_ptr = reinterpret_cast<const Config*>(sorted_i.data_ptr());
        const auto* sorted_pi_ptr = reinterpret_cast<const Coef2*>(sorted_psi_i.data_ptr());
        const auto* bj_ptr = reinterpret_cast<const Config*>(block_j.data_ptr());
        auto num_blocks = dim3{
            static_cast<unsigned int>((term_number + threads_per_block.x - 1) / threads_per_block.x),
            static_cast<unsigned int>((batch_j + threads_per_block.y - 1) / threads_per_block.y),
        };
        apply_within_subspace_in_double_side_kernel_interface<n_qubytes, particle_cut, max_op_number, false>
            <<<num_blocks, threads_per_block, 0, stream>>>(
                term_number, batch_j, batch_i, site_ptr, kind_ptr, coef_ptr, bj_ptr, sorted_pi_ptr, sorted_i_ptr, result_ptr);
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        return result_psi;
    }
}

}  // namespace

#ifndef N_QUBYTES
#define N_QUBYTES 0
#endif
#ifndef PARTICLE_CUT
#define PARTICLE_CUT 0
#endif
#ifndef MAX_OP_NUMBER
#define MAX_OP_NUMBER 0
#endif
#ifndef N_UP
#define N_UP 0
#endif

#if N_QUBYTES != 0
#define QMP_LIBRARY_HELPER(nq, pc, mo, nu) qmp_spin_separated_hamiltonian_##nq##_##pc##_##mo##_##nu
#define QMP_LIBRARY(nq, pc, mo, nu) QMP_LIBRARY_HELPER(nq, pc, mo, nu)

TORCH_LIBRARY_IMPL(QMP_LIBRARY(N_QUBYTES, PARTICLE_CUT, MAX_OP_NUMBER, N_UP), CUDA, m) {
    m.impl("spin_separated_apply_within_subspace_in_double_side",
           apply_within_subspace_in_double_side_interface<N_QUBYTES, PARTICLE_CUT, MAX_OP_NUMBER, N_UP>);
}
#undef QMP_LIBRARY
#undef QMP_LIBRARY_HELPER
#endif
