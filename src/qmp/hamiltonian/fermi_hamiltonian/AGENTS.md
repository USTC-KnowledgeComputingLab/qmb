# Hamiltonian 子系统设计

## 计划中 / 未实现 (TODO)

四个核心操作已完整实现且与 JAX fallback 对拍通过。以下为**已知的设计目标，当前尚未实现**，实现时以此为蓝本 (但需先解决括号内的约束):

1. **apply_within 哈希表跨调用缓存** — `configs_j` 在多轮迭代 (如 Lanczos 内循环) 不变时复用 GPU 哈希表，省去每次重建。当前 `FermiHamiltonian._apply_hash_cache` 是占位字段，从未启用；kernel 每次调用都重建哈希表。(约束: JAX FFI handler 无状态，跨 `ffi_call` 缓存需在 C++ 侧维护静态状态，架构上需谨慎设计。)
2. **find_topk chunked compaction + `global_min_weight` 剪枝** — 分块处理 term (chunking + 排序 + rebuild + 用 K-th 权重剪枝) 以在 term 数极大时降低哈希表内存。当前为单次 kernel、`global_min_weight` 恒为 0 (无剪枝)。find_topk 的**去重正确性已实现**，只差此规模优化。仅在超大 term 数时才有收益。
3. **多节点多卡 shard_map** — configs 按 batch 维分片、Hamiltonian 复制、`all_gather` 跨卡归并。当前仅用 `devices[0]` 单卡。独立大工程，需单独 spec 与 GPU 验证 (详见下文「多节点多卡」)。

> 另有一处**后端行为差异** (非 TODO，是有意设计): JAX fallback 的 `find_all`/`find_topk` 用固定容量数组，超容量时静默丢弃且无信号 (CUDA 路径有 overflow retry)，调用 fallback 时须给足 `hash_capacity`。

## 概述

Hamiltonian 子系统负责量子多体哈密顿量的存储和操作。核心设计围绕三个层面:

1. **预处理层** (Python): 将量子化学哈密顿量 (产生/湮灭算符乘积) 转换为位运算友好的紧凑表示
2. **计算层** (CUDA C++): 四个核心操作 (compute_diagonal_within_subspace, apply_within_subspace, find_all_relative_configs, find_topk_relative_configs) 的 GPU kernel
3. **绑定层** (Python): 通过 JAX FFI 将 CUDA kernel 注册为 XLA custom call

## 设计原则

### 位运算优于循环

每个 Hamiltonian term 被预处理为一组位掩码 (`create_mask`, `annihilate_mask`, `flip_mask`, `parity_mask`, `parity_const`)，使得最内层 kernel 只需 4 次位运算即可完成可作用性检查 + 构型翻转 + JW 符号计算。

这是本子系统最核心的性能决策。原因: 一次 popcount 指令替代了原先逐 site 的 JW parity 循环，将 term 处理从 O(max_op_number) 降到 O(1)。

### JAX FFI → shard_map 兼容

每个 CUDA kernel 通过 `XLA_FFI_DEFINE_HANDLER_SYMBOL` 导出，`jax.ffi.register_ffi_target` 注册，`jax.ffi.ffi_call` 调用。所有 kernel 采用 `vmap_method="broadcast_all"` 模式——kernel 自身处理 batch 维度，避免 XLA 插入 scan 循环。标量参数 (`direction`, `hash_capacity`, `count_selected`) 通过 FFI **attribute** (`.Attr<int64_t>`) 传入，Python 侧作为关键字实参传 Python `int`。

> **多卡 shard_map: 计划中，尚未实现。** 当前 `devices` 参数只取 `devices[0]` 落到单设备 (`jax.device_put`)；没有 `Mesh`/`shard_map`/`all_gather` 代码。多节点多卡是后续独立工作。

### 后端选择: 由设备平台决定

后端 (CUDA kernel vs 纯 JAX fallback) **由目标设备平台决定**，而非静默探测:

- `devices` 解析出目标 `jax.Device`。平台为 `gpu` (CUDA) → 编译并注册 CUDA FFI; 平台为 `cpu` → 直接用 JAX fallback，不触碰 nvcc。
- CUDA 设备下若编译/注册失败则**抛异常**，绝不静默退化到 fallback (那会掉进慢几个数量级、在真实规模跑不动的路径)。
- CUDA kernel 缓冲区声明为 F64/complex128。使用方必须启用 `jax.config.update("jax_enable_x64", True)`，否则 `jnp` 把 float64 截断成 float32，FFI 操作数 dtype 不匹配 (测试通过 `tests/conftest.py` 全局启用)。

### 数值精度

`coef` 为 f64、psi 为 (real, imag) f64 对 (对应 complex128)。CUDA 与 fallback 因 `atomicAdd` 顺序非确定而非 bit-exact，回归测试用 `allclose(rtol=1e-12)`。

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

`_hamiltonian_prepare.py` 中的 `prepare` 函数负责将产生/湮灭算符序列转换为上述表示。:

给定产生算符列表和湮灭算符列表，模拟逐算符作用过程:
1. 对每个算符，计算其对初始构型的约束条件
2. 累加 JW 奇偶性常数部分
3. 更新翻转掩码
4. 检查约束一致性

## 四个核心操作

CUDA kernel 与纯 JAX fallback 两套实现，输出语义一致 (`allclose` 验证)。

### compute_diagonal_within_subspace

对每个 config 累加不改变构型的哈密顿项系数 (对角元)。`flip_mask[t] == 0` 项才对对角有贡献。`FermiHamiltonian` 初始化时预切对角 term 子集 (`_diag_*`)，只把该子集传给 kernel/fallback，跳过 90-99% 非对角 term。CUDA: 一个线程负责一个 config，寄存器累加所有对角 term 后一次性写回 `psi[i]`(对角元互相独立，无 atomicAdd、无 shared-mem 归约)。

### apply_within_subspace

稀疏矩阵乘向量: H · psi_i 投影到 configs_j 张成的子空间。支持 forward/backward 双向 (`direction`，backward 即 H^†，交换 src/dst 且系数取共轭)。CUDA: 总是遍历 `min(B_src, B_dst)` 较小侧、把线性探测哈希表建在被查找的另一侧以最小化线程数 (flip 自逆，两种遍历结果逐位一致)；主 kernel 查表 O(1) 定位并 `atomicAdd`。

### find_all_relative_configs

全部列举新构型 + 去重 + 振幅累加。CUDA: 自定义 CAS 哈希表 (`findall_slot`, wyhash64)，claim-then-probe 插入 (无自旋，避免 SIMT 死锁)，probe 上限 100 触发 overflow → kernel 返回 overflow 标志，Python 层翻倍 `hash_capacity` 重试 (≤8 次)，不丢构型; 两趟 collect 按"规范首槽"归并并发产生的重复 slot，count 与累加振幅精确。排除集用第二哈希表 (`exclude_slot`) O(1) 查询。**JAX fallback 差异**: fallback 用固定 `hash_capacity` 数组，超容量时静默丢弃且无 overflow 信号 (无重试)，调用方须给足容量。

### find_topk_relative_configs

Top-K 选择。CUDA: 容量 2K 的哈希表按构型聚合权重 (`atomicMax`，double 用 CAS-loop 实现)；排除集同样用第二哈希表 O(1) 查询; collect kernel 按"规范首槽"去重 (每 key 只输出一次、取其重复 slot 的 max 权重)，Python 层 `argsort` 取 top-K——保证 top-K 无重复构型，与 fallback 一致。chunked compaction + `global_min_weight` 剪枝为计划中 (spec §3.2.4)。

## 多节点多卡 (计划中)

以下为**设计目标，当前代码尚未实现**。落地需要独立的 spec 与 GPU 验证:

- configs 按 batch 维度 `shard_map` 分片
- Hamiltonian 参数 (create_mask, annihilate_mask 等) 在所有设备上复制
- 每个设备独立执行 FFI kernel，无需跨设备通信
- 全局归并 (如 find_topk_relative_configs 的 top-K) 通过 `jax.lax.all_gather` 实现

当前实现为单设备: `FermiHamiltonian` 仅使用 `devices[0]`。

## 其他子系统引用

各模型 (FCIDUMP, Hubbard, OpenFermion 等) 在 `models/` 子系统中直接构造 `FermiHamiltonian(hamiltonian_dict, n_qubits=..., devices=...)`。不同模型只负责将自身格式转换为 `{term: complex_coefficient}` 字典，其余逻辑全部复用。
