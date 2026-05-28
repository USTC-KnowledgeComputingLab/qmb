#include <torch/extension.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

namespace {

inline std::uint8_t popcount_parity(std::uint8_t byte) {
    return std::popcount(static_cast<unsigned int>(byte)) & 1;
}

template <std::int64_t size>
struct array_less {
    bool operator()(const std::array<std::uint8_t, size>& lhs,
                    const std::array<std::uint8_t, size>& rhs) const {
        for (std::int64_t i = 0; i < size; ++i) {
            if (lhs[i] < rhs[i]) return true;
            if (lhs[i] > rhs[i]) return false;
        }
        return false;
    }
};

inline bool get_bit(const std::uint8_t* data, std::int64_t index) {
    return (data[index >> 3] >> (index & 7)) & 1;
}

inline void set_bit(std::uint8_t* data, std::int64_t index, bool value) {
    if (value)
        data[index >> 3] |= (1 << (index & 7));
    else
        data[index >> 3] &= ~(1 << (index & 7));
}

template <std::int64_t n_qubytes>
std::uint8_t jw_parity(const std::uint8_t* config, std::int64_t site) {
    std::uint8_t p = 0;
    std::int64_t bi = static_cast<std::int64_t>(site >> 3);
    for (std::int64_t b = 0; b < bi; ++b)
        p ^= popcount_parity(config[b]);
    std::uint8_t mask = static_cast<std::uint8_t>((1 << (site & 7)) - 1);
    p ^= popcount_parity(config[bi] & mask);
    return p;
}

template <std::int64_t n_qubytes, std::int64_t particle_cut, std::int64_t max_op_number, bool forward>
std::pair<bool, bool> hamiltonian_apply_kernel(
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
void apply_within_subspace_in_double_side_kernel(
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
        if (less(current_configs, sorted_dst_configs[mid]))
            hi = mid - 1;
        else if (less(sorted_dst_configs[mid], current_configs))
            lo = mid + 1;
        else { found = true; break; }
    }
    if (!found) return;

    std::int8_t sign = parity ? -1 : +1;
    result_psi[mid][0] += sign * (coef[term_index][0] * src_psi[batch_index][0] - coef[term_index][1] * src_psi[batch_index][1]);
    result_psi[mid][1] += sign * (coef[term_index][0] * src_psi[batch_index][1] + coef[term_index][1] * src_psi[batch_index][0]);
}

template <std::int64_t n_qubytes, std::int64_t particle_cut, std::int64_t max_op_number, bool forward>
void apply_within_subspace_in_double_side_kernel_interface(
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
    for (std::int64_t t = 0; t < term_number; ++t) {
        for (std::int64_t b = 0; b < src_batch_size; ++b) {
            apply_within_subspace_in_double_side_kernel<n_qubytes, particle_cut, max_op_number, forward>(
                t, b, dst_batch_size,
                site, kind, coef,
                src_configs, src_psi,
                sorted_dst_configs, result_psi);
        }
    }
}

template <std::int64_t n_qubytes, std::int64_t particle_cut, std::int64_t max_op_number>
auto sort_configs(const torch::Tensor& configs, torch::Tensor& sort_idx) -> torch::Tensor {
    using Config = std::array<std::uint8_t, n_qubytes>;
    std::int64_t n = configs.size(0);
    auto device = configs.device();
    sort_idx = torch::arange(n, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto* idx_ptr = sort_idx.data_ptr<std::int64_t>();
    const auto* cfg_ptr = reinterpret_cast<const Config*>(configs.data_ptr());
    std::sort(idx_ptr, idx_ptr + n, [cfg_ptr](std::int64_t a, std::int64_t b) {
        return array_less<n_qubytes>()(cfg_ptr[a], cfg_ptr[b]);
    });
    return configs.index({sort_idx});
}

template <std::int64_t n_qubytes, std::int64_t particle_cut, std::int64_t max_op_number>
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

    const auto* site_ptr = reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr());
    const auto* kind_ptr = reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr());
    const auto* coef_ptr = reinterpret_cast<const Coef2*>(coef.data_ptr());

    auto result_psi = torch::zeros({batch_j, 2},
        torch::TensorOptions().dtype(torch::kFloat64).device(configs_i.device()));
    auto* result_ptr = reinterpret_cast<Coef2*>(result_psi.data_ptr());

    if (direction == 0) {
        torch::Tensor sort_j_idx;
        auto sorted_j = configs_j_sorted
            ? configs_j
            : sort_configs<n_qubytes, particle_cut, max_op_number>(configs_j, sort_j_idx);

        const auto* sorted_j_ptr = reinterpret_cast<const Config*>(sorted_j.data_ptr());
        const auto* ci_ptr = reinterpret_cast<const Config*>(configs_i.data_ptr());
        const auto* pi_ptr = reinterpret_cast<const Coef2*>(psi_i.data_ptr());

        apply_within_subspace_in_double_side_kernel_interface<n_qubytes, particle_cut, max_op_number, true>(
            term_number, batch_i, batch_j,
            site_ptr, kind_ptr, coef_ptr,
            ci_ptr, pi_ptr,
            sorted_j_ptr, result_ptr);

        if (!configs_j_sorted) {
            auto unsorted = torch::zeros_like(result_psi);
            unsorted.index_put_({sort_j_idx}, result_psi);
            return unsorted;
        }
        return result_psi;
    } else {
        torch::Tensor sort_i_idx;
        auto sorted_i = configs_i_sorted
            ? configs_i
            : sort_configs<n_qubytes, particle_cut, max_op_number>(configs_i, sort_i_idx);

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
        const auto* cj_ptr = reinterpret_cast<const Config*>(configs_j.data_ptr());

        apply_within_subspace_in_double_side_kernel_interface<n_qubytes, particle_cut, max_op_number, false>(
            term_number, batch_j, batch_i,
            site_ptr, kind_ptr, coef_ptr,
            cj_ptr, sorted_pi_ptr,
            sorted_i_ptr, result_ptr);

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

#if N_QUBYTES != 0
#define QMP_LIBRARY_HELPER(nq, pc, mo) qmp_hamiltonian_##nq##_##pc##_##mo
#define QMP_LIBRARY(nq, pc, mo) QMP_LIBRARY_HELPER(nq, pc, mo)

TORCH_LIBRARY_IMPL(QMP_LIBRARY(N_QUBYTES, PARTICLE_CUT, MAX_OP_NUMBER), CPU, m) {
    m.impl("apply_within_subspace_in_double_side",
           apply_within_subspace_in_double_side_interface<N_QUBYTES, PARTICLE_CUT, MAX_OP_NUMBER>);
}
#undef QMP_LIBRARY
#undef QMP_LIBRARY_HELPER
#endif
