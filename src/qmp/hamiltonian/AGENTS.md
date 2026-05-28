# Hamiltonian 子系统设计

## 设备归属

`Hamiltonian` 类持有一个计算设备，在初始化时从 `devices` 参数解析得到。各操作方法将输入张量（`configs_i`、`psi_i`、`configs_j`）移动到 Hamiltonian 的设备上——而非反过来。Hamiltonian 的内部张量（`site`、`kind`、`coef`）也在初始化时放置到该设备上。

## C++ 内核/接口分离

CPU 和 CUDA 后端各自是独立的自包含文件（`_hamiltonian_cpu.cpp` / `_hamiltonian_cuda.cu`），不共享头文件。每个文件内包含自己的工具函数（位操作、JW 奇偶校验、比较器等）和完整的四层架构。

每个操作遵循相同的分层模式，以 `apply_within_subspace_in_double_side` 为例（当前唯一实现的操作）：

1. **`hamiltonian_apply_kernel`**（`device`/`__device__`）——纯计算核心。将单个项的算符作用到组态上。操作原始 `std::array` 指针。不依赖 PyTorch。CPU 和 CUDA 逻辑完全一致。所有操作（apply_within、find_relative、list_relative、diagonal_term）共用此核心。

2. **`{operation}_kernel`**（`device`/`__device__`）——每个 (term_index, batch_index) 对的工作单元。调用 `hamiltonian_apply_kernel`，执行操作特定的逻辑（二分查找、堆插入、字典树插入等），累积结果。同样操作原始指针。

3. **`{operation}_kernel_interface`**（`host`/`__global__`）——遍历所有 term × batch 对，调用上述 kernel。CPU 版为双层 for 循环，CUDA 版为 2D kernel 启动网格（`grid(term_number, batch_size)`）。

4. **`{operation}_interface`**（host，`TORCH_LIBRARY_IMPL`）——PyTorch 集成层。处理张量排序和原始指针提取。调用 kernel interface。

以 `apply_within_subspace_in_double_side` 为例，支持 `forward` 和 `backward` 两种遍历方向：
- **forward**：遍历 term × config_i，施加算符，在 config_j 中二分查找
- **backward**：遍历 term × config_j，施加逆算符，在 config_i 中二分查找

该方向作为 `bool` 模板参数，通过 `if constexpr` 实现编译期分支。此方向选择仅为 `apply_within_subspace_in_double_side` 的特性（其他操作无此需求）。

### CPU / CUDA 差异

| | CPU (`_hamiltonian_cpu.cpp`) | CUDA (`_hamiltonian_cuda.cu`) |
|---|---|---|
| 线程模型 | 串行 `for term × for batch` | `__global__` 2D kernel，每线程一个 (term, config) 对 |
| 排序 | `std::sort` + 索引数组 + `array_less` | `thrust::sort_by_key` on device |
| 结果累加 | 直接 `+=` | `atomicAdd`（两个 double 各一次） |
| 奇偶校验 | 预计算 `constexpr` 256 字节查表 | `__popc(byte) & 1` 硬件指令 |
| 内存 | host memory | device memory，全部指针为 device 侧 |
| 同步 | 无需 | `cudaDeviceSynchronize()` |
| 核心逻辑 | 完全一致 | 完全一致（仅标注和累加方式不同） |

### TORCH_LIBRARY 注册原则

`TORCH_LIBRARY_FRAGMENT`（operator schema 定义）**只在 `_hamiltonian.cpp` 中存在**。当编译 CPU 后端时，`_hamiltonian.cpp` 被包含在源文件列表中，以 `N_QUBYTES != 0` 编译，因此其 `#else` 分支激活 `TORCH_LIBRARY_FRAGMENT`。CPU 和 CUDA 后端各自仅含 `TORCH_LIBRARY_IMPL`（注册对应实现），不重复声明 operator。

声明模块（`device_type="declaration"`）仍以 `N_QUBYTES=0` 编译 `_hamiltonian.cpp`，走 `PYBIND11_MODULE` 路径提供 `prepare` 函数。

这一设计的原因是：同一 operator schema 多次声明会导致 `c10::Dispatcher::registerDef` panic。通过将 fragment 集中在 `_hamiltonian.cpp` 并由 CPU 后端加载（CPU 必然先于 CUDA 加载），保证 schema 只被注册一次。CUDA 后端的 `TORCH_LIBRARY_IMPL` 可以为已存在的 operator 注册 CUDA 实现，或在 operator 不存在时隐式声明。

## 模板参数命名

所有模板参数采用 `lower_case` 命名（如 `forward`、`max_op_number`、`n_qubytes`、`particle_cut`），对应编译期宏名称（`MAX_OP_NUMBER`、`N_QUBYTES`、`PARTICLE_CUT`）。

标准参数顺序为：`n_qubytes`、`particle_cut`、`max_op_number`、`forward`。适用于所有 C++ 模板签名、Python 方法签名、模块名称和缓存 key。其中 `forward` 仅为 `apply_within_subspace_in_double_side` 系列函数使用，其他操作忽略此参数。

## JIT 编译

Hamiltonian 子系统使用 `torch.utils.cpp_extension.load()` 进行 C++/CUDA 内核的 JIT 编译。编译后的模块缓存在 `platformdirs.user_cache_path("qmp", "kclab")` 下。清除缓存：

```bash
rm -rf ~/.cache/qmp
```
