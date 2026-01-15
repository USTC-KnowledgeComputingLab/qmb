#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <torch/extension.h>

namespace qmp_hamiltonian_cuda {

constexpr torch::DeviceType device = torch::kCUDA;

// 构型数据采用位包装（8 sites/byte）。
// 在二进制紧凑排列下，字节流的字典序比较等价于量子态的逻辑比较。
// 该比较器直接用于 thrust::sort 和后续的二分查找。
template<typename T, std::int64_t size>
struct array_less {
    __device__ bool operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
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

// 计算构型的概率权重（振幅模长平方）。
// 在 Top-K 筛选中，以此作为判断重要构型的依据。
template<typename T, std::int64_t size>
struct array_square_greater {
    __device__ T square(const std::array<T, size>& value) const {
        T result = 0;
        for (std::int64_t i = 0; i < size; ++i) {
            result += value[i] * value[i];
        }
        return result;
    }
    __device__ bool operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
        return square(lhs) > square(rhs);
    }
};

// 位操作：通过字节偏移 (index/8) 和位掩码 (index%8) 定位格点。
__device__ bool get_bit(std::uint8_t* data, std::uint8_t index) {
    return ((*data) >> index) & 1;
}

__device__ void set_bit(std::uint8_t* data, std::uint8_t index, bool value) {
    if (value) {
        *data |= (1 << index);
    } else {
        *data &= ~(1 << index);
    }
}

// 核心逻辑：将哈密顿量的一个项 (term) 作用于当前构型。
// 1. 物理校验：检查费米子产生/湮灭算符的合法性（Pauli 不相容原理）。
// 2. In-place 修改：直接在 current_configs 内存上进行位翻转，避免拷贝。
// 3. 符号计算：通过统计目标格点前的粒子总数（Parity），处理费米子交换符号 (-1)^N。
template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__device__ std::pair<bool, bool> hamiltonian_apply_kernel(
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

// 投影计算：计算 H|ψ⟩ 在给定基组 (result_configs) 上的投影。
// 由于 result_configs 预先已排序，线程内直接使用二分查找定位索引。
// 并发冲突：多个源态可能跃迁到同一个目标态，因此必须使用 atomicAdd 累加振幅。
template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__device__ void apply_within_kernel(
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
    atomicAdd(&result_psi[mid][0], sign * (coef[term_index][0] * psi[batch_index][0] - coef[term_index][1] * psi[batch_index][1]));
    atomicAdd(&result_psi[mid][1], sign * (coef[term_index][0] * psi[batch_index][1] + coef[term_index][1] * psi[batch_index][0]));
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__global__ void apply_within_kernel_interface(
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
    std::int64_t term_index = blockIdx.x * blockDim.x + threadIdx.x;
    std::int64_t batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (term_index < term_number && batch_index < batch_size) {
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

    at::cuda::CUDAGuard cuda_device_guard(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    auto policy = thrust::device.on(stream);
    cudaDeviceProp prop;
    AT_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::int64_t max_threads_per_block = prop.maxThreadsPerBlock;

    TORCH_CHECK(configs.device().type() == torch::kCUDA, "configs must be on CUDA.")
    TORCH_CHECK(configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(configs.size(0) == batch_size, "configs batch size must match the provided batch_size.");
    TORCH_CHECK(configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(psi.device().type() == torch::kCUDA, "psi must be on CUDA.")
    TORCH_CHECK(psi.device().index() == device_id, "psi must be on the same device as others.");
    TORCH_CHECK(psi.is_contiguous(), "psi must be contiguous.")
    TORCH_CHECK(psi.dtype() == torch::kFloat64, "psi must be float64.")
    TORCH_CHECK(psi.dim() == 2, "psi must be 2D.")
    TORCH_CHECK(psi.size(0) == batch_size, "psi batch size must match the provided batch_size.");
    TORCH_CHECK(psi.size(1) == 2, "psi must contain 2 elements for each batch.");

    TORCH_CHECK(result_configs.device().type() == torch::kCUDA, "result_configs must be on CUDA.")
    TORCH_CHECK(result_configs.device().index() == device_id, "result_configs must be on the same device as others.");
    TORCH_CHECK(result_configs.is_contiguous(), "result_configs must be contiguous.")
    TORCH_CHECK(result_configs.dtype() == torch::kUInt8, "result_configs must be uint8.")
    TORCH_CHECK(result_configs.dim() == 2, "result_configs must be 2D.")
    TORCH_CHECK(result_configs.size(0) == result_batch_size, "result_configs batch size must match the provided result_batch_size.")
    TORCH_CHECK(result_configs.size(1) == n_qubytes, "result_configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(site.device().type() == torch::kCUDA, "site must be on CUDA.")
    TORCH_CHECK(site.device().index() == device_id, "site must be on the same device as others.");
    TORCH_CHECK(site.is_contiguous(), "site must be contiguous.")
    TORCH_CHECK(site.dtype() == torch::kInt16, "site must be int16.")
    TORCH_CHECK(site.dim() == 2, "site must be 2D.")
    TORCH_CHECK(site.size(0) == term_number, "site size must match the provided term_number.");
    TORCH_CHECK(site.size(1) == max_op_number, "site must match the provided max_op_number.");

    TORCH_CHECK(kind.device().type() == torch::kCUDA, "kind must be on CUDA.")
    TORCH_CHECK(kind.device().index() == device_id, "kind must be on the same device as others.");
    TORCH_CHECK(kind.is_contiguous(), "kind must be contiguous.")
    TORCH_CHECK(kind.dtype() == torch::kUInt8, "kind must be uint8.")
    TORCH_CHECK(kind.dim() == 2, "kind must be 2D.")
    TORCH_CHECK(kind.size(0) == term_number, "kind size must match the provided term_number.");
    TORCH_CHECK(kind.size(1) == max_op_number, "kind must match the provided max_op_number.");

    TORCH_CHECK(coef.device().type() == torch::kCUDA, "coef must be on CUDA.")
    TORCH_CHECK(coef.device().index() == device_id, "coef must be on the same device as others.");
    TORCH_CHECK(coef.is_contiguous(), "coef must be contiguous.")
    TORCH_CHECK(coef.dtype() == torch::kFloat64, "coef must be float64.")
    TORCH_CHECK(coef.dim() == 2, "coef must be 2D.")
    TORCH_CHECK(coef.size(0) == term_number, "coef size must match the provided term_number.");
    TORCH_CHECK(coef.size(1) == 2, "coef must contain 2 elements for each term.");

    auto sorted_result_configs = result_configs.clone(torch::MemoryFormat::Contiguous);
    auto result_sort_index = torch::arange(result_batch_size, torch::TensorOptions().dtype(torch::kInt64).device(device, device_id));
    thrust::sort_by_key(
        policy,
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_result_configs.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_result_configs.data_ptr()) + result_batch_size,
        reinterpret_cast<std::int64_t*>(result_sort_index.data_ptr()),
        array_less<std::uint8_t, n_qubytes>()
    );
    auto sorted_result_psi = torch::zeros({result_batch_size, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(device, device_id));

    auto threads_per_block = dim3{1, max_threads_per_block >> 1}; // I don't know why, but need to divide by 2 to avoid errors
    auto num_blocks =
        dim3{(term_number + threads_per_block.x - 1) / threads_per_block.x, (batch_size + threads_per_block.y - 1) / threads_per_block.y};
    apply_within_kernel_interface<max_op_number, n_qubytes, particle_cut><<<num_blocks, threads_per_block, 0, stream>>>(
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
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto result_psi = torch::zeros_like(sorted_result_psi);
    result_psi.index_put_({result_sort_index}, sorted_result_psi);
    return result_psi;
}

// 自旋锁实现：使用 atomicCAS 获取锁，配合 nanosleep 进行backoff。
__device__ void _mutex_lock(int* mutex) {
    // I don't know why we need to wait for these periods of time, but the examples in the CUDA documentation are written this way.
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nanosleep-example
    unsigned int ns = 8;
    while (atomicCAS(mutex, 0, 1) == 1) {
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
    }
}

__device__ void mutex_lock(int* mutex) {
    _mutex_lock(mutex);
    __threadfence();
}

__device__ void mutex_lock(int* mutex1, int* mutex2) {
    _mutex_lock(mutex1);
    _mutex_lock(mutex2);
    __threadfence();
}

__device__ void _mutex_unlock(int* mutex) {
    atomicExch(mutex, 0);
}

__device__ void mutex_unlock(int* mutex) {
    __threadfence();
    _mutex_unlock(mutex);
}

__device__ void mutex_unlock(int* mutex1, int* mutex2) {
    __threadfence();
    _mutex_unlock(mutex1);
    _mutex_unlock(mutex2);
}

// 并发 Top-K 筛选堆。
// 为了在 GPU 上从海量并发数据中筛选前 K 个最大值，采用细粒度锁策略：
// 不锁定整个堆，而是给每个节点分配独立的 Mutex。
// 当新元素在堆中下沉时，采用“蟹行”锁定：锁定当前节点 -> 锁定子节点 -> 比较/交换 -> 释放当前节点。
template<typename T, typename Less = thrust::less<T>>
__device__ void add_into_heap(T* heap, int* mutex, std::int64_t heap_size, const T& value) {
    auto less = Less();
    std::int64_t index = 0;
    if (less(value, heap[index])) {
    } else {
        mutex_lock(&mutex[index]);
        if (less(value, heap[index])) {
            mutex_unlock(&mutex[index]);
        } else {
            while (true) {
                // Current lock status: only mutex[index] is locked
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
                                // The left child is greater than the value, treat it as if only the right child is present
                                mutex_lock(&mutex[right]);
                                if (less(value, heap[right])) {
                                    mutex_unlock(&mutex[right]);
                                    break;
                                } else {
                                    heap[index] = heap[right];
                                    mutex_unlock(&mutex[index]);
                                    index = right;
                                }
                            }
                        } else {
                            if (less(value, heap[right])) {
                                // The right child is greater than the value, treat it as if only the left child is present
                                mutex_lock(&mutex[left]);
                                if (less(value, heap[left])) {
                                    mutex_unlock(&mutex[left]);
                                    break;
                                } else {
                                    heap[index] = heap[left];
                                    mutex_unlock(&mutex[index]);
                                    index = left;
                                }
                            } else {
                                mutex_lock(&mutex[left], &mutex[right]);
                                if (less(heap[left], heap[right])) {
                                    if (less(value, heap[left])) {
                                        mutex_unlock(&mutex[left], &mutex[right]);
                                        break;
                                    } else {
                                        heap[index] = heap[left];
                                        mutex_unlock(&mutex[index], &mutex[right]);
                                        index = left;
                                    }
                                } else {
                                    if (less(value, heap[right])) {
                                        mutex_unlock(&mutex[left], &mutex[right]);
                                        break;
                                    } else {
                                        heap[index] = heap[right];
                                        mutex_unlock(&mutex[index], &mutex[left]);
                                        index = right;
                                    }
                                }
                            }
                        }
                    } else {
                        // Only the left child is present
                        if (less(value, heap[left])) {
                            break;
                        } else {
                            mutex_lock(&mutex[left]);
                            if (less(value, heap[left])) {
                                mutex_unlock(&mutex[left]);
                                break;
                            } else {
                                heap[index] = heap[left];
                                mutex_unlock(&mutex[index]);
                                index = left;
                            }
                        }
                    }
                } else {
                    if (right_present) {
                        // Only the right child is present
                        if (less(value, heap[right])) {
                            break;
                        } else {
                            mutex_lock(&mutex[right]);
                            if (less(value, heap[right])) {
                                mutex_unlock(&mutex[right]);
                                break;
                            } else {
                                heap[index] = heap[right];
                                mutex_unlock(&mutex[index]);
                                index = right;
                            }
                        }
                    } else {
                        // No children are present
                        break;
                    }
                }
            }
            heap[index] = value;
            mutex_unlock(&mutex[index]);
        }
    }
}

// 比较器：仅比较前 8 字节（double 权重），用于堆的排序和淘汰。
template<typename T, std::int64_t size>
struct array_first_double_less {
    __device__ double first_double(const std::array<T, size + sizeof(double) / sizeof(T)>& value) const {
        double result;
        for (std::int64_t i = 0; i < sizeof(double); ++i) {
            reinterpret_cast<std::uint8_t*>(&result)[i] = reinterpret_cast<const std::uint8_t*>(&value[0])[i];
        }
        return result;
    }

    __device__ bool
    operator()(const std::array<T, size + sizeof(double) / sizeof(T)>& lhs, const std::array<T, size + sizeof(double) / sizeof(T)>& rhs) const {
        return first_double(lhs) < first_double(rhs);
    }
};

// 子空间扩展：
// 1. 产生新构型。
// 2. 二分查找排除掉已存在的构型。
// 3. 计算权重，尝试插入并发堆。
template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__device__ void find_relative_kernel(
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
    int* mutex,
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
    // Currently, the weight is calculated as the probability of the state, but it can be changed to other values in the future.
    // TODO: I should choose a better way to calculate the weight for theoretical reasons.
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
        mutex,
        heap_size,
        value
    );
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__global__ void find_relative_kernel_interface(
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
    int* mutex,
    std::int64_t heap_size
) {
    std::int64_t term_index = blockIdx.x * blockDim.x + threadIdx.x;
    std::int64_t batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (term_index < term_number && batch_index < batch_size) {
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
            /*mutex=*/mutex,
            /*heap_size=*/heap_size
        );
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

    at::cuda::CUDAGuard cuda_device_guard(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    auto policy = thrust::device.on(stream);
    cudaDeviceProp prop;
    AT_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::int64_t max_threads_per_block = prop.maxThreadsPerBlock;

    TORCH_CHECK(configs.device().type() == torch::kCUDA, "configs must be on CUDA.")
    TORCH_CHECK(configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(configs.size(0) == batch_size, "configs batch size must match the provided batch_size.");
    TORCH_CHECK(configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(psi.device().type() == torch::kCUDA, "psi must be on CUDA.")
    TORCH_CHECK(psi.device().index() == device_id, "psi must be on the same device as others.");
    TORCH_CHECK(psi.is_contiguous(), "psi must be contiguous.")
    TORCH_CHECK(psi.dtype() == torch::kFloat64, "psi must be float64.")
    TORCH_CHECK(psi.dim() == 2, "psi must be 2D.")
    TORCH_CHECK(psi.size(0) == batch_size, "psi batch size must match the provided batch_size.");
    TORCH_CHECK(psi.size(1) == 2, "psi must contain 2 elements for each batch.");

    TORCH_CHECK(site.device().type() == torch::kCUDA, "site must be on CUDA.")
    TORCH_CHECK(site.device().index() == device_id, "site must be on the same device as others.");
    TORCH_CHECK(site.is_contiguous(), "site must be contiguous.")
    TORCH_CHECK(site.dtype() == torch::kInt16, "site must be int16.")
    TORCH_CHECK(site.dim() == 2, "site must be 2D.")
    TORCH_CHECK(site.size(0) == term_number, "site size must match the provided term_number.");
    TORCH_CHECK(site.size(1) == max_op_number, "site must match the provided max_op_number.");

    TORCH_CHECK(kind.device().type() == torch::kCUDA, "kind must be on CUDA.")
    TORCH_CHECK(kind.device().index() == device_id, "kind must be on the same device as others.");
    TORCH_CHECK(kind.is_contiguous(), "kind must be contiguous.")
    TORCH_CHECK(kind.dtype() == torch::kUInt8, "kind must be uint8.")
    TORCH_CHECK(kind.dim() == 2, "kind must be 2D.")
    TORCH_CHECK(kind.size(0) == term_number, "kind size must match the provided term_number.");
    TORCH_CHECK(kind.size(1) == max_op_number, "kind must match the provided max_op_number.");

    TORCH_CHECK(coef.device().type() == torch::kCUDA, "coef must be on CUDA.")
    TORCH_CHECK(coef.device().index() == device_id, "coef must be on the same device as others.");
    TORCH_CHECK(coef.is_contiguous(), "coef must be contiguous.")
    TORCH_CHECK(coef.dtype() == torch::kFloat64, "coef must be float64.")
    TORCH_CHECK(coef.dim() == 2, "coef must be 2D.")
    TORCH_CHECK(coef.size(0) == term_number, "coef size must match the provided term_number.");
    TORCH_CHECK(coef.size(1) == 2, "coef must contain 2 elements for each term.");

    TORCH_CHECK(exclude_configs.device().type() == torch::kCUDA, "configs must be on CUDA.")
    TORCH_CHECK(exclude_configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(exclude_configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(exclude_configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(exclude_configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(exclude_configs.size(0) == exclude_size, "configs batch size must match the provided exclude_size.");
    TORCH_CHECK(exclude_configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    auto sorted_exclude_configs = exclude_configs.clone(torch::MemoryFormat::Contiguous);

    thrust::sort(
        policy,
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

    int* mutex;
    AT_CUDA_CHECK(cudaMalloc(&mutex, sizeof(int) * count_selected));
    AT_CUDA_CHECK(cudaMemset(mutex, 0, sizeof(int) * count_selected));
    auto threads_per_block = dim3{1, max_threads_per_block >> 1}; // I don't know why, but need to divide by 2 to avoid errors
    auto num_blocks =
        dim3{(term_number + threads_per_block.x - 1) / threads_per_block.x, (batch_size + threads_per_block.y - 1) / threads_per_block.y};
    find_relative_kernel_interface<max_op_number, n_qubytes, particle_cut><<<num_blocks, threads_per_block, 0, stream>>>(
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
        /*mutex=*/mutex,
        /*heap_size=*/count_selected
    );
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    AT_CUDA_CHECK(cudaFree(mutex));

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

// 256 叉前缀树：用于在 GPU 上进行并行的构型去重和振幅累加。
// 每一层对应构型的一个字节（uint8），叶子节点存储复数振幅。
template<std::int64_t n_qubytes>
struct dictionary_tree {
    using child_t = dictionary_tree<n_qubytes - 1>;
    child_t* children[max_uint8_t];
    smallest_atomic_int exist[max_uint8_t];
    largest_atomic_int nonzero_count;

    __device__ bool add(std::uint8_t* begin, double real, double imag) {
        std::uint8_t index = *begin;
        if (children[index] == nullptr) {
            if (atomicCAS(&exist[index], smallest_atomic_int(0), smallest_atomic_int(1))) {
                while (atomicCAS((largest_atomic_int*)&children[index], largest_atomic_int(0), largest_atomic_int(0)) == 0) {
                }
            } else {
                auto new_child = (child_t*)malloc(sizeof(child_t));
                assert(new_child != nullptr);
                memset(new_child, 0, sizeof(child_t));
                children[index] = new_child;
                __threadfence();
            }
        }
        if (children[index]->add(begin + 1, real, imag)) {
            atomicAdd(&nonzero_count, 1);
            return true;
        } else {
            return false;
        }
    }

    template<std::int64_t n_total_qubytes>
    __device__ void collect(std::uint64_t index, std::array<std::uint8_t, n_total_qubytes>* configs, std::array<double, 2>* psi) {
        std::uint64_t size_counter = 0;
        for (std::int64_t i = 0; i < max_uint8_t; ++i) {
            if (exist[i]) {
                std::uint64_t new_size_counter = size_counter + children[i]->nonzero_count;
                if (new_size_counter > index) {
                    std::uint64_t new_index = index - size_counter;
                    configs[index][n_total_qubytes - n_qubytes] = i;
                    children[i]->collect<n_total_qubytes>(new_index, &configs[size_counter], &psi[size_counter]);
                    if (atomicSub(&children[i]->nonzero_count, 1) == 1) {
                        free(children[i]);
                    };
                    return;
                }
                size_counter = new_size_counter;
            }
        }
    }
};

// Trie 递归终点：叶子节点存储物理振幅。
template<>
struct dictionary_tree<1> {
    double values[max_uint8_t][2];
    smallest_atomic_int exist[max_uint8_t];
    largest_atomic_int nonzero_count;

    __device__ bool add(std::uint8_t* begin, double real, double imag) {
        std::uint8_t index = *begin;
        atomicAdd(&values[index][0], real);
        atomicAdd(&values[index][1], imag);
        if (atomicCAS(&exist[index], smallest_atomic_int(0), smallest_atomic_int(1))) {
            return false;
        } else {
            atomicAdd(&nonzero_count, 1);
            return true;
        }
    }

    template<std::int64_t n_total_qubytes>
    __device__ void collect(std::uint64_t index, std::array<std::uint8_t, n_total_qubytes>* configs, std::array<double, 2>* psi) {
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

// 列出子空间：
// 1. 产生新构型。
// 2. 二分查找排除掉已存在的构型。
// 3. 将新构型和振幅加入前缀树。
template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__device__ void list_relative_kernel(
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
__global__ void list_relative_kernel_interface(
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
    std::int64_t term_index = blockIdx.x * blockDim.x + threadIdx.x;
    std::int64_t batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (term_index < term_number && batch_index < batch_size) {
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

template<std::int64_t n_qubytes>
__global__ void collect_tree_kernel(
    std::uint64_t result_size,
    dictionary_tree<n_qubytes>* result_tree,
    std::array<std::uint8_t, n_qubytes>* configs,
    std::array<double, 2>* psi
) {
    std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < result_size) {
        result_tree->template collect<n_qubytes>(index, configs, psi);
    }
}

bool first_call_to_set_heap_limit = true;

void _set_heap_limit() {
    // 增加 CUDA 动态内存限制以容纳 Trie 节点。
    if (first_call_to_set_heap_limit) {
        size_t expect = 1 << 60;
        size_t actual = 0;
        do {
            cudaDeviceSetLimit(cudaLimitMallocHeapSize, expect);
            cudaDeviceGetLimit(&actual, cudaLimitMallocHeapSize);
            expect >>= 1;
        } while (actual != expect);

        first_call_to_set_heap_limit = false;
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

    at::cuda::CUDAGuard cuda_device_guard(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    auto policy = thrust::device.on(stream);
    cudaDeviceProp prop;
    AT_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::int64_t max_threads_per_block = prop.maxThreadsPerBlock;
    _set_heap_limit();

    TORCH_CHECK(configs.device().type() == torch::kCUDA, "configs must be on CUDA.")
    TORCH_CHECK(configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(configs.size(0) == batch_size, "configs batch size must match the provided batch_size.");
    TORCH_CHECK(configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(psi.device().type() == torch::kCUDA, "psi must be on CUDA.")
    TORCH_CHECK(psi.device().index() == device_id, "psi must be on the same device as others.");
    TORCH_CHECK(psi.is_contiguous(), "psi must be contiguous.")
    TORCH_CHECK(psi.dtype() == torch::kFloat64, "psi must be float64.")
    TORCH_CHECK(psi.dim() == 2, "psi must be 2D.")
    TORCH_CHECK(psi.size(0) == batch_size, "psi batch size must match the provided batch_size.");
    TORCH_CHECK(psi.size(1) == 2, "psi must contain 2 elements for each batch.");

    TORCH_CHECK(site.device().type() == torch::kCUDA, "site must be on CUDA.")
    TORCH_CHECK(site.device().index() == device_id, "site must be on the same device as others.");
    TORCH_CHECK(site.is_contiguous(), "site must be contiguous.")
    TORCH_CHECK(site.dtype() == torch::kInt16, "site must be int16.")
    TORCH_CHECK(site.dim() == 2, "site must be 2D.")
    TORCH_CHECK(site.size(0) == term_number, "site size must match the provided term_number.");
    TORCH_CHECK(site.size(1) == max_op_number, "site must match the provided max_op_number.");

    TORCH_CHECK(kind.device().type() == torch::kCUDA, "kind must be on CUDA.")
    TORCH_CHECK(kind.device().index() == device_id, "kind must be on the same device as others.");
    TORCH_CHECK(kind.is_contiguous(), "kind must be contiguous.")
    TORCH_CHECK(kind.dtype() == torch::kUInt8, "kind must be uint8.")
    TORCH_CHECK(kind.dim() == 2, "kind must be 2D.")
    TORCH_CHECK(kind.size(0) == term_number, "kind size must match the provided term_number.");
    TORCH_CHECK(kind.size(1) == max_op_number, "kind must match the provided max_op_number.");

    TORCH_CHECK(coef.device().type() == torch::kCUDA, "coef must be on CUDA.")
    TORCH_CHECK(coef.device().index() == device_id, "coef must be on the same device as others.");
    TORCH_CHECK(coef.is_contiguous(), "coef must be contiguous.")
    TORCH_CHECK(coef.dtype() == torch::kFloat64, "coef must be float64.")
    TORCH_CHECK(coef.dim() == 2, "coef must be 2D.")
    TORCH_CHECK(coef.size(0) == term_number, "coef size must match the provided term_number.");
    TORCH_CHECK(coef.size(1) == 2, "coef must contain 2 elements for each term.");

    TORCH_CHECK(exclude_configs.device().type() == torch::kCUDA, "configs must be on CUDA.")
    TORCH_CHECK(exclude_configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(exclude_configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(exclude_configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(exclude_configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(exclude_configs.size(0) == exclude_size, "configs batch size must match the provided exclude_size.");
    TORCH_CHECK(exclude_configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    auto sorted_exclude_configs = exclude_configs.clone(torch::MemoryFormat::Contiguous);
    thrust::sort(
        policy,
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr()) + exclude_size,
        array_less<std::uint8_t, n_qubytes>()
    );

    dictionary_tree<n_qubytes>* result_tree;
    AT_CUDA_CHECK(cudaMalloc(&result_tree, sizeof(dictionary_tree<n_qubytes>)));
    AT_CUDA_CHECK(cudaMemset(result_tree, 0, sizeof(dictionary_tree<n_qubytes>)));

    auto threads_per_block = dim3{1, max_threads_per_block >> 1};
    auto num_blocks = dim3{
        (unsigned int)(term_number + threads_per_block.x - 1) / threads_per_block.x,
        (unsigned int)(batch_size + threads_per_block.y - 1) / threads_per_block.y
    };
    list_relative_kernel_interface<max_op_number, n_qubytes, particle_cut><<<num_blocks, threads_per_block, 0, stream>>>(
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

    long long result_size;
    AT_CUDA_CHECK(cudaMemcpyAsync(&result_size, &result_tree->nonzero_count, sizeof(long long), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto result_configs = torch::zeros({result_size, n_qubytes}, torch::TensorOptions().dtype(torch::kUInt8).device(device, device_id));
    auto result_psi = torch::zeros({result_size, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(device, device_id));

    if (result_size > 0) {
        int threads_collect = prop.maxThreadsPerBlock >> 1;
        int blocks_collect = (result_size + threads_collect - 1) / threads_collect;
        collect_tree_kernel<n_qubytes><<<blocks_collect, threads_collect, 0, stream>>>(
            result_size,
            result_tree,
            reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(result_configs.data_ptr()),
            reinterpret_cast<std::array<double, 2>*>(result_psi.data_ptr())
        );
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    AT_CUDA_CHECK(cudaFree(result_tree));
    return std::make_tuple(result_configs, result_psi);
}

// 计算哈密顿量的对角贡献。
// 当哈密顿量项作用后的构型保持不变时，累加其复数系数。
template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__device__ void diagonal_term_kernel(
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
    atomicAdd(&result_psi[batch_index][0], sign * coef[term_index][0]);
    atomicAdd(&result_psi[batch_index][1], sign * coef[term_index][1]);
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__global__ void diagonal_term_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    std::array<double, 2>* result_psi
) {
    std::int64_t term_index = blockIdx.x * blockDim.x + threadIdx.x;
    std::int64_t batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (term_index < term_number && batch_index < batch_size) {
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

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
auto diagonal_term_interface(const torch::Tensor& configs, const torch::Tensor& site, const torch::Tensor& kind, const torch::Tensor& coef)
    -> torch::Tensor {
    std::int64_t device_id = configs.device().index();
    std::int64_t batch_size = configs.size(0);
    std::int64_t term_number = site.size(0);

    at::cuda::CUDAGuard cuda_device_guard(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    auto policy = thrust::device.on(stream);
    cudaDeviceProp prop;
    AT_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::int64_t max_threads_per_block = prop.maxThreadsPerBlock;

    TORCH_CHECK(configs.device().type() == torch::kCUDA, "configs must be on CUDA.")
    TORCH_CHECK(configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(configs.size(0) == batch_size, "configs batch size must match the provided batch_size.");
    TORCH_CHECK(configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(site.device().type() == torch::kCUDA, "site must be on CUDA.")
    TORCH_CHECK(site.device().index() == device_id, "site must be on the same device as others.");
    TORCH_CHECK(site.is_contiguous(), "site must be contiguous.")
    TORCH_CHECK(site.dtype() == torch::kInt16, "site must be int16.")
    TORCH_CHECK(site.dim() == 2, "site must be 2D.")
    TORCH_CHECK(site.size(0) == term_number, "site size must match the provided term_number.");
    TORCH_CHECK(site.size(1) == max_op_number, "site must match the provided max_op_number.");

    TORCH_CHECK(kind.device().type() == torch::kCUDA, "kind must be on CUDA.")
    TORCH_CHECK(kind.device().index() == device_id, "kind must be on the same device as others.");
    TORCH_CHECK(kind.is_contiguous(), "kind must be contiguous.")
    TORCH_CHECK(kind.dtype() == torch::kUInt8, "kind must be uint8.")
    TORCH_CHECK(kind.dim() == 2, "kind must be 2D.")
    TORCH_CHECK(kind.size(0) == term_number, "kind size must match the provided term_number.");
    TORCH_CHECK(kind.size(1) == max_op_number, "kind must match the provided max_op_number.");

    TORCH_CHECK(coef.device().type() == torch::kCUDA, "coef must be on CUDA.")
    TORCH_CHECK(coef.device().index() == device_id, "coef must be on the same device as others.");
    TORCH_CHECK(coef.is_contiguous(), "coef must be contiguous.")
    TORCH_CHECK(coef.dtype() == torch::kFloat64, "coef must be float64.")
    TORCH_CHECK(coef.dim() == 2, "coef must be 2D.")
    TORCH_CHECK(coef.size(0) == term_number, "coef size must match the provided term_number.");
    TORCH_CHECK(coef.size(1) == 2, "coef must contain 2 elements for each term.");

    auto result_psi = torch::zeros({batch_size, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(device, device_id));

    auto threads_per_block = dim3{1, max_threads_per_block >> 1}; // I don't know why, but need to divide by 2 to avoid errors
    auto num_blocks =
        dim3{(term_number + threads_per_block.x - 1) / threads_per_block.x, (batch_size + threads_per_block.y - 1) / threads_per_block.y};
    diagonal_term_kernel_interface<max_op_number, n_qubytes, particle_cut><<<num_blocks, threads_per_block, 0, stream>>>(
        /*term_number=*/term_number,
        /*batch_size=*/batch_size,
        /*site=*/reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr()),
        /*kind=*/reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr()),
        /*coef=*/reinterpret_cast<const std::array<double, 2>*>(coef.data_ptr()),
        /*configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(configs.data_ptr()),
        /*result_psi=*/reinterpret_cast<std::array<double, 2>*>(result_psi.data_ptr())
    );
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

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
TORCH_LIBRARY_IMPL(QMP_LIBRARY(N_QUBYTES, PARTICLE_CUT), CUDA, m) {
    m.impl("apply_within", apply_within_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
    m.impl("find_relative", find_relative_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
    m.impl("list_relative", list_relative_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
    m.impl("diagonal_term", diagonal_term_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
}
#undef QMP_LIBRARY
#undef QMP_LIBRARY_HELPER
#endif

} // namespace qmp_hamiltonian_cuda
