#include <torch/extension.h>

namespace qmp_hamiltonian_cpu {

constexpr torch::DeviceType device = torch::kCPU;

template<typename T, std::int64_t size>
struct array_less {
    bool operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
        for (std::int64_t i = 0; i < size; ++i) {
            if (lhs[i] < rhs[i]) {
                return true;
            }
            if (lhs[i] > rhs[i]) {
                return false;
            }
        }
        return false;
    }
};

template<typename T, std::int64_t size>
struct array_square_greater {
    T square(const std::array<T, size>& value) const {
        T result = 0;
        for (std::int64_t i = 0; i < size; ++i) {
            result += value[i] * value[i];
        }
        return result;
    }
    bool operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
        return square(lhs) > square(rhs);
    }
};

bool get_bit(std::uint8_t* data, std::uint8_t index) {
    return ((*data) >> index) & 1;
}

void set_bit(std::uint8_t* data, std::uint8_t index, bool value) {
    if (value) {
        *data |= (1 << index);
    } else {
        *data &= ~(1 << index);
    }
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
std::pair<bool, bool> hamiltonian_apply_kernel(
    std::array<std::uint8_t, n_qubytes>& current_configs,
    std::int64_t term_index,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind // term_number
) {
    static_assert(particle_cut == 1 || particle_cut == 2, "particle_cut != 1 or 2 not implemented");
    bool success = true;
    bool parity = false;
    for (std::int64_t op_index = max_op_number; op_index-- > 0;) {
        std::int16_t site_single = site[term_index][op_index];
        std::uint8_t kind_single = kind[term_index][op_index];
        if (kind_single == 2) {
            continue;
        }
        std::uint8_t to_what = kind_single;
        if (get_bit(&current_configs[site_single / 8], site_single % 8) == to_what) {
            success = false;
            break;
        }
        set_bit(&current_configs[site_single / 8], site_single % 8, to_what);
        if constexpr (particle_cut == 1) {
            for (std::int16_t s = 0; s < site_single; ++s) {
                parity ^= get_bit(&current_configs[s / 8], s % 8);
            }
        }
    }
    return std::make_pair(success, parity);
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
void apply_within_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t term_number,
    std::int64_t batch_size,
    std::int64_t result_batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    const std::array<std::uint8_t, n_qubytes>* result_configs, // result_batch_size
    std::array<double, 2>* result_psi
) {
    std::array<std::uint8_t, n_qubytes> current_configs = configs[batch_index];
    auto [success, parity] = hamiltonian_apply_kernel<max_op_number, n_qubytes, particle_cut>(
        /*current_configs=*/current_configs,
        /*term_index=*/term_index,
        /*site=*/site,
        /*kind=*/kind
    );

    if (!success) {
        return;
    }
    success = false;
    std::int64_t low = 0;
    std::int64_t high = result_batch_size - 1;
    std::int64_t mid = 0;
    auto less = array_less<std::uint8_t, n_qubytes>();
    while (low <= high) {
        mid = (low + high) / 2;
        if (less(current_configs, result_configs[mid])) {
            high = mid - 1;
        } else if (less(result_configs[mid], current_configs)) {
            low = mid + 1;
        } else {
            success = true;
            break;
        }
    }
    if (!success) {
        return;
    }
    std::int8_t sign = parity ? -1 : +1;
    result_psi[mid][0] += sign * (coef[term_index][0] * psi[batch_index][0] - coef[term_index][1] * psi[batch_index][1]);
    result_psi[mid][1] += sign * (coef[term_index][0] * psi[batch_index][1] + coef[term_index][1] * psi[batch_index][0]);
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
void apply_within_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    std::int64_t result_batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    const std::array<std::uint8_t, n_qubytes>* result_configs, // result_batch_size
    std::array<double, 2>* result_psi
) {
    for (std::int64_t term_index = 0; term_index < term_number; ++term_index) {
        for (std::int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
            apply_within_kernel<max_op_number, n_qubytes, particle_cut>(
                /*term_index=*/term_index,
                /*batch_index=*/batch_index,
                /*term_number=*/term_number,
                /*batch_size=*/batch_size,
                /*result_batch_size=*/result_batch_size,
                /*site=*/site,
                /*kind=*/kind,
                /*coef=*/coef,
                /*configs=*/configs,
                /*psi=*/psi,
                /*result_configs=*/result_configs,
                /*result_psi=*/result_psi
            );
        }
    }
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
auto apply_within_interface(
    const torch::Tensor& configs,
    const torch::Tensor& psi,
    const torch::Tensor& result_configs,
    const torch::Tensor& site,
    const torch::Tensor& kind,
    const torch::Tensor& coef
) -> torch::Tensor {
    std::int64_t device_id = configs.device().index();
    std::int64_t batch_size = configs.size(0);
    std::int64_t result_batch_size = result_configs.size(0);
    std::int64_t term_number = site.size(0);

    TORCH_CHECK(configs.device().type() == torch::kCPU, "configs must be on CPU.")
    TORCH_CHECK(configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(configs.size(0) == batch_size, "configs batch size must match the provided batch_size.");
    TORCH_CHECK(configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(psi.device().type() == torch::kCPU, "psi must be on CPU.")
    TORCH_CHECK(psi.device().index() == device_id, "psi must be on the same device as others.");
    TORCH_CHECK(psi.is_contiguous(), "psi must be contiguous.")
    TORCH_CHECK(psi.dtype() == torch::kFloat64, "psi must be float64.")
    TORCH_CHECK(psi.dim() == 2, "psi must be 2D.")
    TORCH_CHECK(psi.size(0) == batch_size, "psi batch size must match the provided batch_size.");
    TORCH_CHECK(psi.size(1) == 2, "psi must contain 2 elements for each batch.");

    TORCH_CHECK(result_configs.device().type() == torch::kCPU, "result_configs must be on CPU.")
    TORCH_CHECK(result_configs.device().index() == device_id, "result_configs must be on the same device as others.");
    TORCH_CHECK(result_configs.is_contiguous(), "result_configs must be contiguous.")
    TORCH_CHECK(result_configs.dtype() == torch::kUInt8, "result_configs must be uint8.")
    TORCH_CHECK(result_configs.dim() == 2, "result_configs must be 2D.")
    TORCH_CHECK(result_configs.size(0) == result_batch_size, "result_configs batch size must match the provided result_batch_size.")
    TORCH_CHECK(result_configs.size(1) == n_qubytes, "result_configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(site.device().type() == torch::kCPU, "site must be on CPU.")
    TORCH_CHECK(site.device().index() == device_id, "site must be on the same device as others.");
    TORCH_CHECK(site.is_contiguous(), "site must be contiguous.")
    TORCH_CHECK(site.dtype() == torch::kInt16, "site must be int16.")
    TORCH_CHECK(site.dim() == 2, "site must be 2D.")
    TORCH_CHECK(site.size(0) == term_number, "site size must match the provided term_number.");
    TORCH_CHECK(site.size(1) == max_op_number, "site must match the provided max_op_number.");

    TORCH_CHECK(kind.device().type() == torch::kCPU, "kind must be on CPU.")
    TORCH_CHECK(kind.device().index() == device_id, "kind must be on the same device as others.");
    TORCH_CHECK(kind.is_contiguous(), "kind must be contiguous.")
    TORCH_CHECK(kind.dtype() == torch::kUInt8, "kind must be uint8.")
    TORCH_CHECK(kind.dim() == 2, "kind must be 2D.")
    TORCH_CHECK(kind.size(0) == term_number, "kind size must match the provided term_number.");
    TORCH_CHECK(kind.size(1) == max_op_number, "kind must match the provided max_op_number.");

    TORCH_CHECK(coef.device().type() == torch::kCPU, "coef must be on CPU.")
    TORCH_CHECK(coef.device().index() == device_id, "coef must be on the same device as others.");
    TORCH_CHECK(coef.is_contiguous(), "coef must be contiguous.")
    TORCH_CHECK(coef.dtype() == torch::kFloat64, "coef must be float64.")
    TORCH_CHECK(coef.dim() == 2, "coef must be 2D.")
    TORCH_CHECK(coef.size(0) == term_number, "coef size must match the provided term_number.");
    TORCH_CHECK(coef.size(1) == 2, "coef must contain 2 elements for each term.");

    auto result_sort_index = torch::arange(result_batch_size, torch::TensorOptions().dtype(torch::kInt64).device(device, device_id));
    std::sort(
        reinterpret_cast<std::int64_t*>(result_sort_index.data_ptr()),
        reinterpret_cast<std::int64_t*>(result_sort_index.data_ptr()) + result_batch_size,
        [&result_configs](std::int64_t i1, std::int64_t i2) {
            return array_less<std::uint8_t, n_qubytes>()(
                reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(result_configs.data_ptr())[i1],
                reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(result_configs.data_ptr())[i2]
            );
        }
    );
    auto sorted_result_configs = result_configs.index({result_sort_index});
    auto sorted_result_psi = torch::zeros({result_batch_size, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(device, device_id));

    apply_within_kernel_interface<max_op_number, n_qubytes, particle_cut>(
        /*term_number=*/term_number,
        /*batch_size=*/batch_size,
        /*result_batch_size=*/result_batch_size,
        /*site=*/reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr()),
        /*kind=*/reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr()),
        /*coef=*/reinterpret_cast<const std::array<double, 2>*>(coef.data_ptr()),
        /*configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(configs.data_ptr()),
        /*psi=*/reinterpret_cast<const std::array<double, 2>*>(psi.data_ptr()),
        /*result_configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(sorted_result_configs.data_ptr()),
        /*result_psi=*/reinterpret_cast<std::array<double, 2>*>(sorted_result_psi.data_ptr())
    );

    auto result_psi = torch::zeros_like(sorted_result_psi);
    result_psi.index_put_({result_sort_index}, sorted_result_psi);
    return result_psi;
}

template<typename T, typename Less = std::less<T>>
void add_into_heap(T* heap, std::int64_t heap_size, const T& value) {
    auto less = Less();
    std::int64_t index = 0;
    if (less(value, heap[index])) {
    } else {
        while (true) {
            // Calculate the indices of the left and right children
            std::int64_t left = (index << 1) + 1;
            std::int64_t right = (index << 1) + 2;
            std::int64_t left_present = left < heap_size;
            std::int64_t right_present = right < heap_size;
            if (left_present) {
                if (right_present) {
                    // Both left and right children are present
                    if (less(value, heap[left])) {
                        if (less(value, heap[right])) {
                            // Both children are greater than the value, break
                            break;
                        } else {
                            // The left child is greater than the value
                            heap[index] = heap[right];
                            index = right;
                        }
                    } else {
                        if (less(value, heap[right])) {
                            // The right child is greater than the value
                            heap[index] = heap[left];
                            index = left;
                        } else {
                            if (less(heap[left], heap[right])) {
                                heap[index] = heap[left];
                                index = left;
                            } else {
                                heap[index] = heap[right];
                                index = right;
                            }
                        }
                    }
                } else {
                    // Only the left child is present
                    if (less(value, heap[left])) {
                        break;
                    } else {
                        heap[index] = heap[left];
                        index = left;
                    }
                }
            } else {
                if (right_present) {
                    // Only the right child is present
                    if (less(value, heap[right])) {
                        break;
                    } else {
                        heap[index] = heap[right];
                        index = right;
                    }
                } else {
                    // No children are present
                    break;
                }
            }
        }
        heap[index] = value;
    }
}

template<typename T, std::int64_t size>
struct array_first_double_less {
    double first_double(const std::array<T, size + sizeof(double) / sizeof(T)>& value) const {
        double result;
        for (std::int64_t i = 0; i < sizeof(double); ++i) {
            reinterpret_cast<std::uint8_t*>(&result)[i] = reinterpret_cast<const std::uint8_t*>(&value[0])[i];
        }
        return result;
    }

    bool operator()(const std::array<T, size + sizeof(double) / sizeof(T)>& lhs, const std::array<T, size + sizeof(double) / sizeof(T)>& rhs) const {
        return first_double(lhs) < first_double(rhs);
    }
};

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
void find_relative_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t term_number,
    std::int64_t batch_size,
    std::int64_t exclude_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    const std::array<std::uint8_t, n_qubytes>* exclude_configs, // exclude_size
    std::array<std::uint8_t, n_qubytes + sizeof(double) / sizeof(std::uint8_t)>* heap,
    std::int64_t heap_size
) {
    std::array<std::uint8_t, n_qubytes> current_configs = configs[batch_index];
    auto [success, parity] = hamiltonian_apply_kernel<max_op_number, n_qubytes, particle_cut>(
        /*current_configs=*/current_configs,
        /*term_index=*/term_index,
        /*site=*/site,
        /*kind=*/kind
    );

    if (!success) {
        return;
    }
    success = true;
    std::int64_t low = 0;
    std::int64_t high = exclude_size - 1;
    std::int64_t mid = 0;
    auto less = array_less<std::uint8_t, n_qubytes>();
    while (low <= high) {
        mid = (low + high) / 2;
        if (less(current_configs, exclude_configs[mid])) {
            high = mid - 1;
        } else if (less(exclude_configs[mid], current_configs)) {
            low = mid + 1;
        } else {
            success = false;
            break;
        }
    }
    if (!success) {
        return;
    }
    std::int8_t sign = parity ? -1 : +1;
    double real = sign * (coef[term_index][0] * psi[batch_index][0] - coef[term_index][1] * psi[batch_index][1]);
    double imag = sign * (coef[term_index][0] * psi[batch_index][1] + coef[term_index][1] * psi[batch_index][0]);
    double weight = real * real + imag * imag;
    std::array<std::uint8_t, n_qubytes + sizeof(double) / sizeof(std::uint8_t)> value;
    for (std::int64_t i = 0; i < sizeof(double) / sizeof(uint8_t); ++i) {
        value[i] = reinterpret_cast<const std::uint8_t*>(&weight)[i];
    }
    for (std::int64_t i = 0; i < n_qubytes; ++i) {
        value[i + sizeof(double) / sizeof(uint8_t)] = current_configs[i];
    }
    add_into_heap<std::array<std::uint8_t, n_qubytes + sizeof(double) / sizeof(std::uint8_t)>, array_first_double_less<std::uint8_t, n_qubytes>>(
        heap,
        heap_size,
        value
    );
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
void find_relative_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    std::int64_t exclude_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    const std::array<std::uint8_t, n_qubytes>* exclude_configs, // exclude_size
    std::array<std::uint8_t, n_qubytes + sizeof(double) / sizeof(std::uint8_t)>* heap,
    std::int64_t heap_size
) {
    for (std::int64_t term_index = 0; term_index < term_number; ++term_index) {
        for (std::int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
            find_relative_kernel<max_op_number, n_qubytes, particle_cut>(
                /*term_index=*/term_index,
                /*batch_index=*/batch_index,
                /*term_number=*/term_number,
                /*batch_size=*/batch_size,
                /*exclude_size=*/exclude_size,
                /*site=*/site,
                /*kind=*/kind,
                /*coef=*/coef,
                /*configs=*/configs,
                /*psi=*/psi,
                /*exclude_configs=*/exclude_configs,
                /*heap=*/heap,
                /*heap_size=*/heap_size
            );
        }
    }
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
auto find_relative_interface(
    const torch::Tensor& configs,
    const torch::Tensor& psi,
    const std::int64_t count_selected,
    const torch::Tensor& site,
    const torch::Tensor& kind,
    const torch::Tensor& coef,
    const torch::Tensor& exclude_configs
) -> torch::Tensor {
    std::int64_t device_id = configs.device().index();
    std::int64_t batch_size = configs.size(0);
    std::int64_t term_number = site.size(0);
    std::int64_t exclude_size = exclude_configs.size(0);

    TORCH_CHECK(configs.device().type() == torch::kCPU, "configs must be on CPU.")
    TORCH_CHECK(configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(configs.size(0) == batch_size, "configs batch size must match the provided batch_size.");
    TORCH_CHECK(configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(psi.device().type() == torch::kCPU, "psi must be on CPU.")
    TORCH_CHECK(psi.device().index() == device_id, "psi must be on the same device as others.");
    TORCH_CHECK(psi.is_contiguous(), "psi must be contiguous.")
    TORCH_CHECK(psi.dtype() == torch::kFloat64, "psi must be float64.")
    TORCH_CHECK(psi.dim() == 2, "psi must be 2D.")
    TORCH_CHECK(psi.size(0) == batch_size, "psi batch size must match the provided batch_size.");
    TORCH_CHECK(psi.size(1) == 2, "psi must contain 2 elements for each batch.");

    TORCH_CHECK(site.device().type() == torch::kCPU, "site must be on CPU.")
    TORCH_CHECK(site.device().index() == device_id, "site must be on the same device as others.");
    TORCH_CHECK(site.is_contiguous(), "site must be contiguous.")
    TORCH_CHECK(site.dtype() == torch::kInt16, "site must be int16.")
    TORCH_CHECK(site.dim() == 2, "site must be 2D.")
    TORCH_CHECK(site.size(0) == term_number, "site size must match the provided term_number.");
    TORCH_CHECK(site.size(1) == max_op_number, "site must match the provided max_op_number.");

    TORCH_CHECK(kind.device().type() == torch::kCPU, "kind must be on CPU.")
    TORCH_CHECK(kind.device().index() == device_id, "kind must be on the same device as others.");
    TORCH_CHECK(kind.is_contiguous(), "kind must be contiguous.")
    TORCH_CHECK(kind.dtype() == torch::kUInt8, "kind must be uint8.")
    TORCH_CHECK(kind.dim() == 2, "kind must be 2D.")
    TORCH_CHECK(kind.size(0) == term_number, "kind size must match the provided term_number.");
    TORCH_CHECK(kind.size(1) == max_op_number, "kind must match the provided max_op_number.");

    TORCH_CHECK(coef.device().type() == torch::kCPU, "coef must be on CPU.")
    TORCH_CHECK(coef.device().index() == device_id, "coef must be on the same device as others.");
    TORCH_CHECK(coef.is_contiguous(), "coef must be contiguous.")
    TORCH_CHECK(coef.dtype() == torch::kFloat64, "coef must be float64.")
    TORCH_CHECK(coef.dim() == 2, "coef must be 2D.")
    TORCH_CHECK(coef.size(0) == term_number, "coef size must match the provided term_number.");
    TORCH_CHECK(coef.size(1) == 2, "coef must contain 2 elements for each term.");

    TORCH_CHECK(exclude_configs.device().type() == torch::kCPU, "configs must be on CPU.")
    TORCH_CHECK(exclude_configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(exclude_configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(exclude_configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(exclude_configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(exclude_configs.size(0) == exclude_size, "configs batch size must match the provided exclude_size.");
    TORCH_CHECK(exclude_configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    auto sorted_exclude_configs = exclude_configs.clone(torch::MemoryFormat::Contiguous);

    std::sort(
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr()) + exclude_size,
        array_less<std::uint8_t, n_qubytes>()
    );

    auto result_pool = torch::zeros(
        {count_selected, n_qubytes + sizeof(double) / sizeof(std::uint8_t)},
        torch::TensorOptions().dtype(torch::kUInt8).device(device, device_id)
    );
    std::array<std::uint8_t, n_qubytes + sizeof(double) / sizeof(std::uint8_t)>* heap =
        reinterpret_cast<std::array<std::uint8_t, n_qubytes + sizeof(double) / sizeof(std::uint8_t)>*>(result_pool.data_ptr());

    find_relative_kernel_interface<max_op_number, n_qubytes, particle_cut>(
        /*term_number=*/term_number,
        /*batch_size=*/batch_size,
        /*exclude_size=*/exclude_size,
        /*site=*/reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr()),
        /*kind=*/reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr()),
        /*coef=*/reinterpret_cast<const std::array<double, 2>*>(coef.data_ptr()),
        /*configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(configs.data_ptr()),
        /*psi=*/reinterpret_cast<const std::array<double, 2>*>(psi.data_ptr()),
        /*exclude_configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr()),
        /*heap=*/heap,
        /*heap_size=*/count_selected
    );

    // Here, the bytes before sizeof(double) / sizeof(std::uint8_t) in result_pool are weights, and the bytes after are configs
    // We need to remove items with weight 0, then sort and deduplicate the configs

    auto result_config =
        result_pool.index({torch::indexing::Slice(), torch::indexing::Slice(sizeof(double) / sizeof(std::uint8_t), torch::indexing::None)});
    auto result_weight =
        result_pool.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, sizeof(double) / sizeof(std::uint8_t))});
    auto nonzero = torch::any(result_weight != 0, /*dim=*/1);
    auto nonzero_result_config = result_config.index({nonzero});
    auto unique_nonzero_result_config =
        std::get<0>(torch::unique_dim(/*self=*/nonzero_result_config, /*dim=*/0, /*sorted=*/true, /*return_inverse=*/false, /*return_counts=*/false));
    return unique_nonzero_result_config;
}

constexpr std::int64_t max_uint8_t = 256;
using largest_atomic_int = unsigned int; // The largest int type that can be atomicAdd/atomicSub
using smallest_atomic_int = unsigned short int; // The smallest int type that can be atomicCAS

template<std::int64_t n_qubytes>
struct dictionary_tree {
    using child_t = dictionary_tree<n_qubytes - 1>;
    child_t* children[max_uint8_t];
    smallest_atomic_int exist[max_uint8_t];
    largest_atomic_int nonzero_count;

    bool add(const std::uint8_t* begin, double real, double imag) {
        std::uint8_t index = *begin;
        if (children[index] == nullptr) {
            auto new_child = (child_t*)malloc(sizeof(child_t));
            assert(new_child != nullptr);
            memset(new_child, 0, sizeof(child_t));
            children[index] = new_child;
            exist[index] = 1;
        }

        if (children[index]->add(begin + 1, real, imag)) {
            nonzero_count++;
            return true;
        } else {
            return false;
        }
    }

    template<std::int64_t n_total_qubytes>
    void collect(std::uint64_t index, std::array<std::uint8_t, n_total_qubytes>* configs, std::array<double, 2>* psi) {
        std::uint64_t size_counter = 0;
        for (std::int64_t i = 0; i < max_uint8_t; ++i) {
            if (exist[i]) {
                std::uint64_t new_size_counter = size_counter + children[i]->nonzero_count;
                if (new_size_counter > index) {
                    std::uint64_t new_index = index - size_counter;
                    configs[index][n_total_qubytes - n_qubytes] = i;
                    children[i]->collect<n_total_qubytes>(new_index, &configs[size_counter], &psi[size_counter]);
                    if (--children[i]->nonzero_count == 0) {
                        free(children[i]);
                    };
                    return;
                }
                size_counter = new_size_counter;
            }
        }
    }
};

template<>
struct dictionary_tree<1> {
    double values[max_uint8_t][2];
    smallest_atomic_int exist[max_uint8_t];
    largest_atomic_int nonzero_count;

    bool add(const std::uint8_t* begin, double real, double imag) {
        std::uint8_t index = *begin;
        values[index][0] += real;
        values[index][1] += imag;
        if (exist[index] == 0) {
            exist[index] = 1;
            ++nonzero_count;
            return true;
        } else {
            return false;
        }
    }

    template<std::int64_t n_total_qubytes>
    void collect(std::uint64_t index, std::array<std::uint8_t, n_total_qubytes>* configs, std::array<double, 2>* psi) {
        std::uint64_t size_counter = 0;
        for (std::int64_t i = 0; i < max_uint8_t; ++i) {
            if (exist[i]) {
                if (size_counter == index) {
                    configs[index][n_total_qubytes - 1] = i;
                    psi[index][0] = values[i][0];
                    psi[index][1] = values[i][1];
                    return;
                }
                ++size_counter;
            }
        }
    }
};

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
void list_relative_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t term_number,
    std::int64_t batch_size,
    std::int64_t exclude_size,
    const std::array<std::int16_t, max_op_number>* site,
    const std::array<std::uint8_t, max_op_number>* kind,
    const std::array<double, 2>* coef,
    const std::array<std::uint8_t, n_qubytes>* configs,
    const std::array<double, 2>* psi,
    const std::array<std::uint8_t, n_qubytes>* exclude_configs,
    dictionary_tree<n_qubytes>* result_tree
) {
    std::array<std::uint8_t, n_qubytes> current_configs = configs[batch_index];
    auto [success, parity] = hamiltonian_apply_kernel<max_op_number, n_qubytes, particle_cut>(
        /*current_configs=*/current_configs,
        /*term_index=*/term_index,
        /*site=*/site,
        /*kind=*/kind
    );

    if (!success) {
        return;
    }

    // 二分查找：排除掉已经在 exclude_configs 中的基矢
    std::int64_t low = 0, high = exclude_size - 1;
    auto compare = array_less<std::uint8_t, n_qubytes>();
    while (low <= high) {
        std::int64_t mid = (low + high) / 2;
        if (compare(current_configs, exclude_configs[mid])) {
            high = mid - 1;
        } else if (compare(exclude_configs[mid], current_configs)) {
            low = mid + 1;
        } else {
            return; // 构型已存在
        }
    }

    std::int8_t sign = parity ? -1 : +1;
    result_tree->add(
        current_configs.data(),
        sign * (coef[term_index][0] * psi[batch_index][0] - coef[term_index][1] * psi[batch_index][1]),
        sign * (coef[term_index][0] * psi[batch_index][1] + coef[term_index][1] * psi[batch_index][0])
    );
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
void list_relative_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    std::int64_t exclude_size,
    const std::array<std::int16_t, max_op_number>* site,
    const std::array<std::uint8_t, max_op_number>* kind,
    const std::array<double, 2>* coef,
    const std::array<std::uint8_t, n_qubytes>* configs,
    const std::array<double, 2>* psi,
    const std::array<std::uint8_t, n_qubytes>* exclude_configs,
    dictionary_tree<n_qubytes>* result_tree
) {
    for (std::int64_t term_index = 0; term_index < term_number; ++term_index) {
        for (std::int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
            list_relative_kernel<max_op_number, n_qubytes, particle_cut>(
                term_index,
                batch_index,
                term_number,
                batch_size,
                exclude_size,
                site,
                kind,
                coef,
                configs,
                psi,
                exclude_configs,
                result_tree
            );
        }
    }
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
auto list_relative_interface(
    const torch::Tensor& configs,
    const torch::Tensor& psi,
    const torch::Tensor& site,
    const torch::Tensor& kind,
    const torch::Tensor& coef,
    const torch::Tensor& exclude_configs
) -> std::tuple<torch::Tensor, torch::Tensor> {
    std::int64_t device_id = configs.device().index();
    std::int64_t batch_size = configs.size(0);
    std::int64_t term_number = site.size(0);
    std::int64_t exclude_size = exclude_configs.size(0);

    TORCH_CHECK(configs.device().type() == torch::kCPU, "configs must be on CPU.")
    TORCH_CHECK(configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(configs.size(0) == batch_size, "configs batch size must match the provided batch_size.");
    TORCH_CHECK(configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(psi.device().type() == torch::kCPU, "psi must be on CPU.")
    TORCH_CHECK(psi.device().index() == device_id, "psi must be on the same device as others.");
    TORCH_CHECK(psi.is_contiguous(), "psi must be contiguous.")
    TORCH_CHECK(psi.dtype() == torch::kFloat64, "psi must be float64.")
    TORCH_CHECK(psi.dim() == 2, "psi must be 2D.")
    TORCH_CHECK(psi.size(0) == batch_size, "psi batch size must match the provided batch_size.");
    TORCH_CHECK(psi.size(1) == 2, "psi must contain 2 elements for each batch.");

    TORCH_CHECK(site.device().type() == torch::kCPU, "site must be on CPU.")
    TORCH_CHECK(site.device().index() == device_id, "site must be on the same device as others.");
    TORCH_CHECK(site.is_contiguous(), "site must be contiguous.")
    TORCH_CHECK(site.dtype() == torch::kInt16, "site must be int16.")
    TORCH_CHECK(site.dim() == 2, "site must be 2D.")
    TORCH_CHECK(site.size(0) == term_number, "site size must match the provided term_number.");
    TORCH_CHECK(site.size(1) == max_op_number, "site must match the provided max_op_number.");

    TORCH_CHECK(kind.device().type() == torch::kCPU, "kind must be on CPU.")
    TORCH_CHECK(kind.device().index() == device_id, "kind must be on the same device as others.");
    TORCH_CHECK(kind.is_contiguous(), "kind must be contiguous.")
    TORCH_CHECK(kind.dtype() == torch::kUInt8, "kind must be uint8.")
    TORCH_CHECK(kind.dim() == 2, "kind must be 2D.")
    TORCH_CHECK(kind.size(0) == term_number, "kind size must match the provided term_number.");
    TORCH_CHECK(kind.size(1) == max_op_number, "kind must match the provided max_op_number.");

    TORCH_CHECK(coef.device().type() == torch::kCPU, "coef must be on CPU.")
    TORCH_CHECK(coef.device().index() == device_id, "coef must be on the same device as others.");
    TORCH_CHECK(coef.is_contiguous(), "coef must be contiguous.")
    TORCH_CHECK(coef.dtype() == torch::kFloat64, "coef must be float64.")
    TORCH_CHECK(coef.dim() == 2, "coef must be 2D.")
    TORCH_CHECK(coef.size(0) == term_number, "coef size must match the provided term_number.");
    TORCH_CHECK(coef.size(1) == 2, "coef must contain 2 elements for each term.");

    TORCH_CHECK(exclude_configs.device().type() == torch::kCPU, "configs must be on CPU.")
    TORCH_CHECK(exclude_configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(exclude_configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(exclude_configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(exclude_configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(exclude_configs.size(0) == exclude_size, "configs batch size must match the provided exclude_size.");
    TORCH_CHECK(exclude_configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    auto sorted_exclude_configs = exclude_configs.clone(torch::MemoryFormat::Contiguous);
    std::sort(
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr()) + exclude_size,
        array_less<std::uint8_t, n_qubytes>()
    );

    auto result_tree = (dictionary_tree<n_qubytes>*)malloc(sizeof(dictionary_tree<n_qubytes>));
    assert(result_tree != nullptr);
    memset(result_tree, 0, sizeof(dictionary_tree<n_qubytes>));

    list_relative_kernel_interface<max_op_number, n_qubytes, particle_cut>(
        term_number,
        batch_size,
        exclude_size,
        reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr()),
        reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr()),
        reinterpret_cast<const std::array<double, 2>*>(coef.data_ptr()),
        reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(configs.data_ptr()),
        reinterpret_cast<const std::array<double, 2>*>(psi.data_ptr()),
        reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr()),
        result_tree
    );

    long long result_size = result_tree->nonzero_count;

    auto result_configs = torch::zeros({result_size, n_qubytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    auto result_psi = torch::zeros({result_size, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    for (std::int64_t i = 0; i < result_size; ++i) {
        result_tree->template collect<n_qubytes>(
            i,
            reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(result_configs.data_ptr()),
            reinterpret_cast<std::array<double, 2>*>(result_psi.data_ptr())
        );
    }

    free(result_tree);
    return std::make_tuple(result_configs, result_psi);
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
void diagonal_term_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t term_number,
    std::int64_t batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    std::array<double, 2>* result_psi
) {
    std::array<std::uint8_t, n_qubytes> current_configs = configs[batch_index];
    auto [success, parity] = hamiltonian_apply_kernel<max_op_number, n_qubytes, particle_cut>(
        /*current_configs=*/current_configs,
        /*term_index=*/term_index,
        /*site=*/site,
        /*kind=*/kind
    );

    if (!success) {
        return;
    }
    auto less = array_less<std::uint8_t, n_qubytes>();
    if (less(current_configs, configs[batch_index]) || less(configs[batch_index], current_configs)) {
        return; // The term does not apply to the current configuration
    }
    std::int8_t sign = parity ? -1 : +1;
    result_psi[batch_index][0] += sign * coef[term_index][0];
    result_psi[batch_index][1] += sign * coef[term_index][1];
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
void diagonal_term_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    std::array<double, 2>* result_psi
) {
    for (std::int64_t term_index = 0; term_index < term_number; ++term_index) {
        for (std::int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
            diagonal_term_kernel<max_op_number, n_qubytes, particle_cut>(
                /*term_index=*/term_index,
                /*batch_index=*/batch_index,
                /*term_number=*/term_number,
                /*batch_size=*/batch_size,
                /*site=*/site,
                /*kind=*/kind,
                /*coef=*/coef,
                /*configs=*/configs,
                /*result_psi=*/result_psi
            );
        }
    }
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
auto diagonal_term_interface(const torch::Tensor& configs, const torch::Tensor& site, const torch::Tensor& kind, const torch::Tensor& coef)
    -> torch::Tensor {
    std::int64_t device_id = configs.device().index();
    std::int64_t batch_size = configs.size(0);
    std::int64_t term_number = site.size(0);

    TORCH_CHECK(configs.device().type() == torch::kCPU, "configs must be on CPU.")
    TORCH_CHECK(configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(configs.size(0) == batch_size, "configs batch size must match the provided batch_size.");
    TORCH_CHECK(configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(site.device().type() == torch::kCPU, "site must be on CPU.")
    TORCH_CHECK(site.device().index() == device_id, "site must be on the same device as others.");
    TORCH_CHECK(site.is_contiguous(), "site must be contiguous.")
    TORCH_CHECK(site.dtype() == torch::kInt16, "site must be int16.")
    TORCH_CHECK(site.dim() == 2, "site must be 2D.")
    TORCH_CHECK(site.size(0) == term_number, "site size must match the provided term_number.");
    TORCH_CHECK(site.size(1) == max_op_number, "site must match the provided max_op_number.");

    TORCH_CHECK(kind.device().type() == torch::kCPU, "kind must be on CPU.")
    TORCH_CHECK(kind.device().index() == device_id, "kind must be on the same device as others.");
    TORCH_CHECK(kind.is_contiguous(), "kind must be contiguous.")
    TORCH_CHECK(kind.dtype() == torch::kUInt8, "kind must be uint8.")
    TORCH_CHECK(kind.dim() == 2, "kind must be 2D.")
    TORCH_CHECK(kind.size(0) == term_number, "kind size must match the provided term_number.");
    TORCH_CHECK(kind.size(1) == max_op_number, "kind must match the provided max_op_number.");

    TORCH_CHECK(coef.device().type() == torch::kCPU, "coef must be on CPU.")
    TORCH_CHECK(coef.device().index() == device_id, "coef must be on the same device as others.");
    TORCH_CHECK(coef.is_contiguous(), "coef must be contiguous.")
    TORCH_CHECK(coef.dtype() == torch::kFloat64, "coef must be float64.")
    TORCH_CHECK(coef.dim() == 2, "coef must be 2D.")
    TORCH_CHECK(coef.size(0) == term_number, "coef size must match the provided term_number.");
    TORCH_CHECK(coef.size(1) == 2, "coef must contain 2 elements for each term.");

    auto result_psi = torch::zeros({batch_size, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(device, device_id));

    diagonal_term_kernel_interface<max_op_number, n_qubytes, particle_cut>(
        /*term_number=*/term_number,
        /*batch_size=*/batch_size,
        /*site=*/reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr()),
        /*kind=*/reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr()),
        /*coef=*/reinterpret_cast<const std::array<double, 2>*>(coef.data_ptr()),
        /*configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(configs.data_ptr()),
        /*result_psi=*/reinterpret_cast<std::array<double, 2>*>(result_psi.data_ptr())
    );

    return result_psi;
}

#ifndef N_QUBYTES
#define N_QUBYTES 0
#endif
#ifndef PARTICLE_CUT
#define PARTICLE_CUT 0
#endif

#if N_QUBYTES != 0
#define QMP_LIBRARY_HELPER(x, y) qmp_hamiltonian_##x##_##y
#define QMP_LIBRARY(x, y) QMP_LIBRARY_HELPER(x, y)
TORCH_LIBRARY_IMPL(QMP_LIBRARY(N_QUBYTES, PARTICLE_CUT), CPU, m) {
    m.impl("apply_within", apply_within_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
    m.impl("find_relative", find_relative_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
    m.impl("list_relative", list_relative_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
    m.impl("diagonal_term", diagonal_term_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
}
#undef QMP_LIBRARY
#undef QMP_LIBRARY_HELPER
#endif

} // namespace qmp_hamiltonian_cpu
