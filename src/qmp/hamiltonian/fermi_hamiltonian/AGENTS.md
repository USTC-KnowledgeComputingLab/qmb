# Hamiltonian 子系统设计

## 概述

Hamiltonian 子系统负责量子多体哈密顿量的存储和操作。核心设计围绕三个层面:

1. **预处理层** (Python + C++): 将量子化学哈密顿量 (产生/湮灭算符乘积) 转换为位运算友好的紧凑表示
2. **计算层** (CUDA C++): 四个核心操作 (diagonal_term, apply_within, list_relative, find_relative) 的 GPU kernel
3. **绑定层** (Python): 通过 JAX FFI 将 CUDA kernel 注册为 XLA custom call

## 设计原则

### 位运算优于循环

每个 Hamiltonian term 被预处理为一组位掩码 (`create_mask`, `annihilate_mask`, `flip_mask`, `parity_mask`, `parity_const`)，使得最内层 kernel 只需 4 次位运算即可完成可作用性检查 + 构型翻转 + JW 符号计算。

这是本子系统最核心的性能决策。原因: 一次 popcount 指令替代了原先逐 site 的 JW parity 循环，将 term 处理从 O(max_op_number) 降到 O(1)。

### JAX FFI → shard_map 兼容

每个 CUDA kernel 通过 `XLA_FFI_DEFINE_HANDLER_SYMBOL` 导出，`jax.ffi.register_ffi_target` 注册，`jax.ffi.ffi_call` 调用。所有 kernel 采用 `vmap_method="broadcast_all"` 模式——kernel 自身处理 batch 维度，避免 XLA 插入 scan 循环。

`shard_map` 分发逻辑: configs 按 batch 维度分片，Hamiltonian 参数 (`create_mask` 等) 在 mesh 上复制 (`P(None)`)。每个设备独立执行 FFI kernel，无需跨设备通信。

### CPU/CUDA 对称性

CPU 和 CUDA 两个后端的核心逻辑尽可能保持一致。平台差异局限于设备管理、kernel 启动和同步方式。

## 位运算表示

### 参数

每个 fermionic term 由六个参数描述:

| 参数 | 形状 | dtype | 含义 |
|------|------|-------|------|
| `create_mask` | [T, Q] | uint8 | 必须为 0 的位 (产生算符目标) |
| `annihilate_mask` | [T, Q] | uint8 | 必须为 1 的位 (湮灭算符目标) |
| `flip_mask` | [T, Q] | uint8 | 翻转掩码 (作用后 XOR) |
| `parity_mask` | [T, Q] | uint8 | JW 奇偶性掩码 |
| `parity_const` | [T] | uint8 | 固定奇偶性 bit (0/1) |
| `coef` | [T, 2] | f64 | 复数系数 (real, imag) |

其中 T = term 数量，Q = n_qubytes = ceil(n_qubits/8)。

### 核心运算伪代码

```
applicable = (config & create_mask[t]) == 0
          && (config & annihilate_mask[t]) == annihilate_mask[t]
new_config = config ^ flip_mask[t]
parity     = parity_const[t] ^ (popcount(parity_mask[t] & config) & 1)
```

### 预处理

`_hamiltonian.cpp` 中的 `prepare` 函数负责将产生/湮灭算符序列转换为上述表示。:

给定产生算符列表和湮灭算符列表，模拟逐算符作用过程:
1. 对每个算符，计算其对初始构型的约束条件
2. 累加 JW 奇偶性常数部分
3. 更新翻转掩码
4. 检查约束一致性

## 四个核心操作

### compute_diagonal_within_subspace

对每个 config 累加不改变构型的哈密顿项系数 (对角元)。`t == 0` 项才对对角有贡献。

### apply_within_subspace

稀疏矩阵乘向量: H · psi_i 投影到 configs_j 张成的子空间。支持 forward/backward 双向遍历。使用 cuCollections `cuco::static_map` 哈希表进行 O(1) config 查找——二分查找在 GPU 上因 warp divergence 不可行。

### find_all_relative_configs

全部列举新构型 + 去重 + 振幅累加。使用 cuCollections `cuco::static_map` 预分配哈希表。Bloom filter 预筛查排除已知构型。

### find_topk_relative_configs

Top-K 选择。使用哈希表 + CUB radix sort 最终排序。

## 多节点多卡

- configs 按 batch 维度 `shard_map` 分片
- Hamiltonian 参数 (create_mask, annihilate_mask 等) 在所有设备上复制
- 每个设备独立执行 FFI kernel，无需跨设备通信
- 全局归并 (如 find_relative 的 top-K) 通过 `jax.lax.all_gather` 实现

## 其他子系统引用

各模型 (FCIDUMP, Hubbard 等) 的 Hamiltonian 通过 `Hamiltonian.from_fcidump()` 等工厂方法构建，返回统一的 `Hamiltonian` 实例。不同模型只负责将自身格式转换为 `{term: complex_coefficient}` 字典，其余逻辑全部复用。
