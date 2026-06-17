# qmp-kit: PyTorch → JAX 迁移设计文档

## 1. 背景与动机

将 qmp-kit 从 PyTorch 迁移到 JAX，同时保留并优化现有的手写 CUDA kernel。

**核心原因**:
- JAX 的 SPMD 编程模型 (`shard_map`, GSPMD) 更适合多节点多卡并行
- JAX 的函数式编程 (`jit`, `grad`, `vmap`) 天然适合 VMC/SCI 优化循环
- 多节点数据并行是 JAX 的一等公民，无需额外框架 (DDP/FSDP)
- 网络层 (MLP/Transformers/MPS) 在 JAX 生态 (Flax/Haiku/Equinox) 中表现优异
- Hamiltonian 层不需要 autodiff 集成 (FFI 即可)

**规模目标**:
- Configs: 10^7 量级
- Hamiltonian terms: 10^5 量级
- Qubits: 100-200
- 多节点多卡，数据并行在 configs 维度

## 2. 项目约定

继承 `refact` 分支已确立的工程规范:

### 2.1 构建与工具

- **构建后端**: hatchling + hatch-vcs
- **包布局**: src layout (`src/qmp/`)
- **License**: AGPL-3.0-only
- **Python**: ≥3.13
- **依赖管理**: uv
- **格式化 + Lint**: ruff (E, F, I, W, UP, SIM, C4, B, RUF, N, T20, PLC)
- **类型检查**: ty (所有函数签名和非平凡 method 必须有类型注解)
- **字符串**: 双引号 (`"`)，行长度 120，导入按 isort 规则排序

### 2.2 设计原则 (来自 AGENTS.md)

- **纯函数优于副作用**: 输出通过返回值传递，不通过可变引用参数
- **文档是交付物**: spec、plan、AGENTS.md 必须在代码稳定后统一检查并更新
- **参考旧代码**: `old/` 中的 main 分支代码是历经迭代验证的参考实现
- **设计先于实现**: 任何非平凡的改动先产出 spec 和 plan，再动手写代码
- **性能选择需有可解释的理由**: 不是"这样更快"，而是"这样更快，因为..."

### 2.3 目录结构

```
qmp/
├── pyproject.toml          # hatchling + hatch-vcs
├── AGENTS.md               # 开发指南
├── LICENSE.md              # AGPL-3.0-only
├── src/qmp/                # 源码 (src layout)
│   ├── __init__.py
│   ├── _version.py         # hatch-vcs 自动生成
│   ├── version.py
│   ├── hamiltonian/        # CUDA kernel + Python thin wrapper
│   ├── networks/           # MLP / Transformers / MPS (→ Flax)
│   ├── algorithms/         # HAAR / VMC / Lanczos (→ 纯 JAX)
│   ├── models/             # FCIDUMP / Hubbard / Ising / PySCF / OpenFermion
│   ├── plugins/            # 第三方框架接口
│   └── utility/            # bitspack, losses, context, optimizer
├── tests/
├── docs/superpowers/
│   ├── specs/              # 设计 spec
│   └── plans/              # 实施 plan
└── old/                    # main 分支参考代码
```

## 3. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│  JAX Python Layer (multi-process SPMD via jax.distributed)  │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │  ANSATZ       │  │  Lanczos      │  │  Config         │  │
│  │  (Flax/纯JAX) │  │  (纯JAX)      │  │  Extension      │  │
│  │               │  │               │  │  (orchestration)│  │
│  └───────────────┘  └───────────────┘  └──────┬─────────┘  │
│                                               │             │
│  ┌────────────────────────────────────────────▼──────────┐  │
│  │         shard_map / custom_partitioning               │  │
│  │    (跨 GPU/节点的 SPMD 分发, 无显式通信代码)           │  │
│  └──────┬───────────────────────────────────────────────┘  │
│         │                                                  │
│  ┌──────▼───────────────────────────────────────────────┐  │
│  │  JAX FFI (jax.extend.ffi)                            │  │
│  │  CUDA kernel 通过 XLA custom call 接入               │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │ diagonal_term │ apply_within │ find_relative     │ │  │
│  │  │ (不变)        │ (二级索引)    │ (哈希表+compact)  │ │  │
│  │  │               │              │ list_relative     │ │  │
│  │  │               │              │ (cuCollections)   │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 JAX FFI 集成方式

调研结论：**JAX FFI (`jax.ffi`) 是唯一正确选择。** Pallas 缺少原子操作支持，无法实现 hash-table 去重。DLPack 只能作为临时迁移桥接，不应是最终方案。CuPy 无法进入 `jax.jit`。

每个 Hamiltonian 操作通过 `jax.ffi` 注册为 XLA custom call:

```python
import jax
import jax.ffi
import ctypes

# 加载编译好的 .so
_lib = ctypes.cdll.LoadLibrary("libqmp_hamiltonian.so")

# 注册每个 kernel 的 FFI target
jax.ffi.register_ffi_target(
    "qmp_diagonal_term",
    jax.ffi.pycapsule(_lib.DiagonalTerm),
    platform="CUDA",
)

# 包装为 JAX callable
def diagonal_term(configs, create_mask, annihilate_mask, flip_mask,
                  parity_mask, parity_const, coef):
    B = configs.shape[0]
    return jax.ffi.ffi_call(
        "qmp_diagonal_term",
        jax.ShapeDtypeStruct((B, 2), jnp.float64),
        vmap_method="broadcast_all",
    )(configs, create_mask, annihilate_mask, flip_mask,
      parity_mask, parity_const, coef)

# 与 shard_map 配合 (多卡)
from jax.sharding import PartitionSpec as P

@partial(jax.shard_map, mesh=mesh,
         in_specs=(P('configs'), P(None), P(None), P(None), P(None), P(None), P(None)),
         out_specs=P('configs'))
def distributed_diagonal_term(configs_shard, create_mask, annihilate_mask,
                               flip_mask, parity_mask, parity_const, coef):
    return diagonal_term(configs_shard, create_mask, annihilate_mask,
                         flip_mask, parity_mask, parity_const, coef)
```

**C++ 侧 handler 模式** (每个 kernel 一个 handler):

```cpp
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

ffi::Error DiagonalTermImpl(
    cudaStream_t stream,                         // FFI 自动注入
    ffi::Buffer<ffi::U8>  configs,               // [B, Q]
    ffi::Buffer<ffi::U8>  create_mask,           // [T, Q]
    ffi::Buffer<ffi::U8>  annihilate_mask,       // [T, Q]
    ffi::Buffer<ffi::U8>  flip_mask,             // [T, Q]
    ffi::Buffer<ffi::U8>  parity_mask,           // [T, Q]
    ffi::Buffer<ffi::U8>  parity_const,          // [T]
    ffi::Buffer<ffi::F64> coef,                  // [T, 2]
    ffi::ResultBuffer<ffi::F64> psi_result       // [B, 2]
) {
    int64_t B = configs.dimensions()[0];
    int64_t T = create_mask.dimensions()[0];
    int64_t Q = create_mask.dimensions()[1];

    cudaMemsetAsync(psi_result->untyped_data(), 0,
                    psi_result->size_bytes(), stream);

    // 启动 CUDA kernel
    diagonal_term_kernel<<<grid, block, 0, stream>>>(
        B, T, Q,
        configs.typed_data(), create_mask.typed_data(),
        annihilate_mask.typed_data(), flip_mask.typed_data(),
        parity_mask.typed_data(), parity_const.typed_data(),
        reinterpret_cast<const double*>(coef.typed_data()),
        reinterpret_cast<double*>(psi_result->typed_data()));

    return ffi::Error::Success();
}

// 导出为 C ABI 符号
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DiagonalTerm, DiagonalTermImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::U8>>()    // configs
        .Arg<ffi::Buffer<ffi::U8>>()    // create_mask
        .Arg<ffi::Buffer<ffi::U8>>()    // annihilate_mask
        .Arg<ffi::Buffer<ffi::U8>>()    // flip_mask
        .Arg<ffi::Buffer<ffi::U8>>()    // parity_mask
        .Arg<ffi::Buffer<ffi::U8>>()    // parity_const
        .Arg<ffi::Buffer<ffi::F64>>()   // coef
        .Ret<ffi::Buffer<ffi::F64>>()   // psi_result
);
```

**编译** (CMake):

```cmake
find_path(XLA_INCLUDE_DIR xla/ffi/api/ffi.h
  HINTS "${Python3_SITELIB}/jaxlib/include")

add_library(qmp_hamiltonian SHARED
  diagonal_term.cu apply_within.cu list_relative.cu find_relative.cu)
target_include_directories(qmp_hamiltonian PRIVATE ${XLA_INCLUDE_DIR})
target_link_libraries(qmp_hamiltonian PRIVATE cudart cuco)
set_target_properties(qmp_hamiltonian PROPERTIES
  CUDA_ARCHITECTURES "80;86;89;90")
```

**关键 API 要点**:
- `jax.extend.ffi` 已废弃，使用 `jax.ffi` (JAX v0.4.31+ stable)
- `ffi::Buffer<T>` 提供 `.typed_data()` (返回 T*) 和 `.dimensions()`
- `ffi::ResultBuffer<T>` 用于输出，通过 `->typed_data()` 写入
- `.Ctx<ffi::PlatformStream<cudaStream_t>>()` 自动注入 CUDA stream
- `vmap_method="broadcast_all"` 比 `"sequential"` 快: kernel 本来就处理 batch
- **不需要 `custom_vjp`**: Hamiltonian 操作不参与神经网络梯度
- **动态输出形状**: list_relative/find_relative 预分配 max capacity + 返回 count

### 3.2 迁移策略

1. **网络层**: MLP/Transformers/MPS 算法逻辑不变，用 Flax 重写，`torch.jit.script` → `jax.jit`
2. **算法层**: HAAR/VMC/Lanczos 用纯 JAX 重写，`jax.grad` 替代手写 closure
3. **Hamiltonian 层**: CUDA kernel 通过 JAX FFI 接入，binding 层从 `torch.utils.cpp_extension` 改为 `xla/ffi/api/ffi.h`

## 4. Hamiltonian 层详细设计

### 4.0 Hamiltonian term 的位运算表示 (BIT.md 方案)

每个 fermionic Hamiltonian term 预处理为六个参数 `(create_mask, annihilate_mask, flip_mask, parity_mask, parity_const, coef)`:

| 参数 | 含义 | 用途 |
|------|------|------|
| `create_mask` | 必须为 0 的位（产生算符的目标位） | 可作用性判断 |
| `annihilate_mask` | 必须为 1 的位（湮灭算符的目标位） | 可作用性判断 |
| `flip_mask` | 翻转掩码（所有算符作用的位） | 构型更新 |
| `parity_mask` | JW 奇偶性掩码 | 符号计算 |
| `parity_const` | 固定奇偶性贡献 (0 或 1) | 符号计算 |
| `coef` | 复数系数 (real + imag) | 振幅累加 |

**可作用性判断**（2 次位运算 + 2 次比较）:
```
applicable = ((config & create_mask) == 0)
          && ((config & annihilate_mask) == annihilate_mask)
```
- `create_mask` 中为 1 的位，config 必须为 0（产生算符目标为空）
- `annihilate_mask` 中为 1 的位，config 必须为 1（湮灭算符目标为占据）

**构型更新**（1 次位运算）:
```
new_config = config ^ flip_mask
```

**JW 奇偶性**（1 次 popcount + 1 次 XOR）:
```
parity = parity_const ^ (popcount(parity_mask & config) & 1)
sign = 1 - 2 * parity  // +1 or -1
```

**预处理**: 给定产生/湮灭算符的有序序列，通过 `term_from_normal_ordered` 生成:

```
对每个算符 (idx, is_creation):
  1. flip_bit = (flip >> idx) & 1
     required_init = flip_bit ^ (0 if creation else 1)
     若与已有条件冲突 → 此项恒零
  2. parity_const ^= popcount(flip & low_mask(idx)) % 2
  3. flip ^= (1 << idx)

最终:
  create_mask = {i | cond[i] == 0}
  annihilate_mask = {i | cond[i] == 1}
  flip_mask = flip
  parity_mask = XOR_{k} low_mask(idx_k)
  parity_const = 累加常数
```

**与当前表示 (site, kind, coef) 的对比**:

| | 当前 | 新方案 |
|---|---|---|
| 可作用性检查 | for 循环按算符逐个检查 | 2 次位运算 |
| 构型修改 | for 循环逐位 set_bit | 1 次 XOR |
| JW 符号 | for 循环逐位 parity | 1 次 popcount |
| 每个 term 的代价 | O(max_op_number) | O(1) |
| 数据量 | site[T,4]int16 + kind[T,4]uint8 + coef[T,2]f64 | create_mask[T,Q] + annihilate_mask[T,Q] + flip_mask[T,Q] + parity_mask[T,Q] + parity_const[T] + coef[T,2] |

每个 term 从 ~14 bytes 变为 ~(4*Q + 17) bytes。对于 Q ≈ 25 qubytes (200 qubits)，~117 bytes/term。100K terms ≈ 11.7 MB，可接受。

### 4.1 CUDA kernel 与 JAX 的连接方式

调研结论：**JAX FFI (`jax.ffi`) 是唯一正确选择。** 理由:

| 方案 | 可行性 | 原因 |
|------|--------|------|
| **JAX FFI** | 唯一选择 | production-ready, zero-copy, stream-aware, jit/shard_map 全兼容 |
| DLPack bridge | 仅迁移桥接 | 需要保留 PyTorch 依赖，无法进入 `jax.jit`，多节点不可行 |
| Pallas | 不可行 | 缺少 `atomicAdd`/`atomicCAS` 支持，无法实现哈希表去重 |
| CuPy interop | 不可行 | 无法在 `jax.jit` 内调用 |

**核心路径**: CUDA `.cu` → CMake 编译为 `.so` → `XLA_FFI_DEFINE_HANDLER_SYMBOL` 导出 → `jax.ffi.register_ffi_target` 注册 → `jax.ffi.ffi_call` 调用 → `shard_map` 多卡分发。

**关键集成要点**:
- `ffi::Buffer<T>`: `.typed_data()` 获取设备指针，`.dimensions()` 获取形状
- `ffi::PlatformStream<cudaStream_t>`: 自动注入 CUDA stream，kernel 在正确流上启动
- `vmap_method="broadcast_all"`: kernel 本身支持 batch，避免 scan 开销
- 输出动态形状: 预分配 max capacity + 返回 count (list_relative 和 find_relative)
- **不需要 `custom_vjp`**: Hamiltonian 操作不参与神经网络梯度

### 4.2 diagonal_term

**操作**: 对每个 config，累加所有不改变该 config 的哈密顿项系数，得到对角元能量。

**位运算实现**:

```
for each (term_t, config_i):
  // 可作用性检查
  if ((config_i & create_mask[t]) != 0) continue
  if ((config_i & annihilate_mask[t]) != annihilate_mask[t]) continue
  // 对角条件：无净翻转
  if (flip_mask[t] != 0) continue
  // JW 符号
  sign = 1 - 2 * (parity_const[t] ^ (popcount(parity_mask[t] & config_i) & 1))
  // 累加
  psi_result[i] += sign * coef[t]
```

**分析**: 每个 (term, config) 对完全独立。只需两层循环 + 位运算 + popcount，无排序无去重。

**方案**: 纯 CUDA kernel，`(create_mask, annihilate_mask, flip_mask, parity_mask, parity_const, coef)` 为输入。2D grid 不变。

```
FFI 签名:
diagonal_term(
  configs          [B, Q] uint8,
  create_mask      [T, Q] uint8,
  annihilate_mask  [T, Q] uint8,
  flip_mask        [T, Q] uint8,
  parity_mask      [T, Q] uint8,
  parity_const     [T]    uint8,
  coef             [T, 2] f64
) → psi [B, 2] f64 (real + imag)
```

**多卡方案**: configs 按 batch 维度分片，每卡持有完整 Hamiltonian 副本，独立计算，结果天然分片。

### 4.3 apply_within

**操作**: 稀疏矩阵乘向量。输入 configs_i (源空间) + psi_i + configs_j (目标空间)，计算 H·ψ_i 投影到目标空间的结果 ψ_j。

**双向支持** (refact 分支):

| 方向 | 遍历模式 | 算符施加 | 二分查找 | 复杂度 |
|------|---------|---------|---------|--------|
| `forward` | term × config_i | 正常 | 在 config_j 中查找 | O(T × B_i × log B_j) |
| `backward` | term × config_j | 逆算符 | 在 config_i 中查找 | O(T × B_j × log B_i) |

选择 B 较小的一侧遍历以最小化总线程数。

**位运算实现** (forward 方向):

```
for each (term_t, config_i):
  // 可作用性检查 (2 次位运算)
  if ((config_i & create_mask[t]) != 0) continue
  if ((config_i & annihilate_mask[t]) != annihilate_mask[t]) continue
  // 构型翻转 (1 次 XOR)
  new_config = config_i ^ flip_mask[t]
  // JW 符号 (1 次 popcount)
  parity = parity_const[t] ^ (popcount(parity_mask[t] & config_i) & 1)
  sign = 1 - 2 * parity
  contribution = sign * coef[t] * psi_i[i]
  // 二级索引二分查找
  idx = two_level_search(sorted_configs_j, new_config)
  if idx >= 0: atomicAdd(psi_result[idx], contribution)
```

backward 方向: 遍历 term × config_j，施加逆算符，在 sorted_configs_i 中查找。

**改进**: 二级索引二分查找，将全局内存访问从 ~24 次降到 ~2 次。

```
FFI 签名:
apply_within(
  configs_i        [B_i, Q] uint8,
  psi_i            [B_i, 2] f64,
  configs_j        [B_j, Q] uint8,
  create_mask      [T, Q] uint8,
  annihilate_mask  [T, Q] uint8,
  flip_mask        [T, Q] uint8,
  parity_mask      [T, Q] uint8,
  parity_const     [T]    uint8,
  coef             [T, 2] f64,
  direction        int (0=forward, 1=backward)
) → psi_j [B_j, 2] f64
```

**多卡方案**: 源 configs 按 batch 维度分片，每卡独立计算贡献。

### 4.4 list_relative

**操作**: 全部列举 + 去重 + 振幅累加。对每个 (term, config_i) 施加 H 项得到新 config，排除已知 configs，去重并累加相同 config 的振幅。返回所有不重复的新 config 及其累加振幅。

**规模**: 预期 distinct new configs ≈ 10^7-10^8。

**当前实现**: 256叉前缀树 (Trie)，device malloc 分配节点，atomicCAS 锁创建。

**问题**: Trie 每 distinct entry 占用 ~25KB，10^8 entries 需 2.5 TB 显存——不可行。

**方案**: **预分配哈希表 (cuCollections `cuco::static_map`)**

**位运算内核**:

```
for each (term_t, config_i):
  // 可作用性检查 (2 次位运算)
  if ((config_i & create_mask[t]) != 0) continue
  if ((config_i & annihilate_mask[t]) != annihilate_mask[t]) continue
  // 构型翻转 (1 次 XOR)
  new_config = config_i ^ flip_mask[t]
  // 排除已知构型
  if bloom_maybe_present(exclude_set, new_config):
    if binary_search_exact(exclude_configs, new_config): continue
  // JW 符号 + 振幅 (1 次 popcount)
  parity = parity_const[t] ^ (popcount(parity_mask[t] & config_i) & 1)
  sign = 1 - 2 * parity
  contribution = sign * coef[t] * psi_i[i]
  // 哈希表插入
  slot = hash_lookup(new_config)
  if slot.found: atomicAdd(slot.value, contribution)
  else: atomicCAS_claim_slot_and_write(new_config, contribution)
// 收集: 线性扫描哈希表 → 所有非空 slot → (configs, psi)
```

**与 Trie 的对比**:

| | Trie | Hash Table |
|---|---|---|
| 每 entry 内存 | ~25KB | ~48B |
| 10^8 entries | 2.5 TB 不可行 | 4.8 GB 可行 |
| 分配方式 | device malloc (慢) | 预分配 |
| 插入延迟 | CAS + malloc + 递归 | CAS + probe |
| 缓存友好性 | 指针追踪 | 顺序 probing |

**退路 (如果 distinct 超过预期)**:
- 多卡哈希分区: `hash(config) % N_GPUs` 决定归属，`jax.lax.ppermute` 路由
- 或: 分批处理，批间 sort-merge 去重

```
FFI 签名:
list_relative(
  configs_i        [B_i, Q] uint8,
  psi_i            [B_i, 2] f64,
  configs_exclude  [E, Q] uint8,
  create_mask      [T, Q] uint8,
  annihilate_mask  [T, Q] uint8,
  flip_mask        [T, Q] uint8,
  parity_mask      [T, Q] uint8,
  parity_const     [T]    uint8,
  coef             [T, 2] f64,
  hash_capacity    int (预估 distinct / 0.6)
) → (new_configs [hash_capacity, Q] uint8, psi_j [hash_capacity, 2] f64, count int)
```

### 4.5 find_relative

**操作**: 流式 Top-K 选择。对每个 (term, config_i) 施加 H 项得到候选 (new_config, weight)，排除已知 configs，选出最重要的 K 个不重复新 config。K ≈ 100K (十万量级)。

与 `list_relative` 的区别: 不返回所有新 config，只选 Top-K；重要性度量是 `|H_ij * psi_i|²` (per-path contribution squared)，而非累加振幅。

**当前实现**: 2D grid → 并发最小堆 (per-node mutex + nanosleep backoff)

**方案**: **全局哈希表 + 阈值加速 + 周期性 compact**

位运算内核:

```
for each (term_t, config_i):
  // 可作用性检查 (2 次位运算)
  if ((config_i & create_mask[t]) != 0) continue
  if ((config_i & annihilate_mask[t]) != annihilate_mask[t]) continue
  // 构型翻转 (1 次 XOR)
  new_config = config_i ^ flip_mask[t]
  // 阈值快速拒绝 (~2 cycles)
  weight = |coef[t] * psi_i[i]|²
  if weight <= global_min_weight: continue
  // 排除已知构型
  if bloom_maybe_present(exclude_set, new_config):
    if binary_search_exact(exclude_configs, new_config): continue
  // 哈希表插入
  slot = hash_lookup(new_config)
  if slot.found: atomicMax(slot.weight, weight)
  else: atomicCAS_claim_and_write(new_config, weight)

// 周期性 compact:
if load_factor > 0.6:
  CUB::sort_descending(all_entries, by=weight)
  keep top K unique, rebuild hash table
  update global_min_weight  // ~0.3ms per compact
```

**Compact 频率分析**:

```
最坏情况:
  distinct 总数: 10^8
  哈希表可容纳: 200K → compact 后剩 100K → 腾出 100K
  compact 次数: 10^8 / 10^5 = 1000 次
  总 compact 成本: 1000 × 0.3ms = 300ms ✓

实际情况远好于最坏:
  前几次 compact 后，threshold 已逼近第 K 大权重
  此后 99.9%+ 条目直接被阈值过滤
  哈希表很少再满 → compact 频率指数下降
```

**选择哈希表而非分片堆的理由**:
1. K=100K 时堆有 17 层，单线程插入需 ~8 次锁操作；哈希表只需 ~2 次 probe
2. 堆的 per-node mutex 带来锁争用；哈希表的 open addressing probe 是无锁的 (仅 CAS on slot)
3. 堆需要预先确定"每个分片放多少条目"——对未知权重分布不鲁棒；哈希表通过 compact 动态适应
4. 实现复杂度: 哈希表主要用现有库 (cuCollections)；堆需要手写 warp-level 协作逻辑

```
FFI 签名:
find_relative(
  configs_i        [B_i, Q] uint8,
  psi_i            [B_i, 2] f64,
  count_selected   int (K),
  configs_exclude  [E, Q] uint8,
  create_mask      [T, Q] uint8,
  annihilate_mask  [T, Q] uint8,
  flip_mask        [T, Q] uint8,
  parity_mask      [T, Q] uint8,
  parity_const     [T]    uint8,
  coef             [T, 2] f64
) → new_configs [K, Q] uint8
```

## 5. 多节点多卡并行方案

### 5.1 数据分布

```
Configs 按 batch 维度分片到各 GPU:
  - 每个 GPU 持有 configs 的一个分片 + Hamiltonian 的完整副本
  - 每个 GPU 独立进行 H|ψ⟩ 操作 (FFI kernel)
  - 结果天然按 configs 维度分片

全局同步点 (通过 JAX collectives):
  - Lanczos 正交化: jax.lax.psum (AllReduce) for inner products
  - 能量期望值: jax.lax.psum(ε_local) → ε_global
  - find_relative 全局归并: jax.lax.all_gather(各卡top-K) → 全局归并 → 全局top-K
```

### 5.2 实现模式

```python
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# 多节点初始化
jax.distributed.initialize(coordinator_address, num_processes, process_id)

# 跨节点 mesh
devices = mesh_utils.create_device_mesh((num_nodes, gpus_per_node))
mesh = Mesh(devices, ('nodes', 'gpus'))

# 每个操作的 sharding spec
@jax.shard_map(mesh=mesh, in_specs=(P('nodes', 'gpus'), ...), out_specs=P('nodes', 'gpus'))
def apply_within_distributed(configs_i, psi_i, configs_j, site, kind, coef):
    return apply_within_ffi(configs_i, psi_i, configs_j, site, kind, coef)
```

### 5.3 通信开销分析

| 操作 | 通信模式 | 通信量 |
|------|---------|--------|
| apply_within/diagonal_term | 无通信 (数据天然分片) | 0 |
| list_relative | all_gather (仅当跨卡去重) | O(N_distinct × 48B) |
| find_relative final merge | all_gather (每卡top-K) | O(N_gpus × K × 48B) ~ MB 级 |
| Lanczos inner products | psum (scalar) | O(1) |
| 梯度同步 (网络训练) | psum (parameters) | O(params) |

## 6. 网络层迁移

### 6.1 MLP

当前 `torch.nn.Sequential(Linear, SiLU, ...)` + `select_linear_layer` (FakeLinear for zero-dim input) → Flax `nn.Sequential([nn.Dense, nn.silu, ...])`。自回归采样用 `jax.lax.scan` 实现。

### 6.2 Transformers

当前手写 SelfAttention + MoE (DeepSeekMoE: SharedExpert + RoutedExpert + SelectedExpert) → Flax `nn.MultiHeadDotProductAttention` + 自定义 MoE 层。KV-cache 用 `jax.lax.scan` 的 carry 实现。

### 6.3 MPS

当前 `torch.nn.Parameter(sites, physical_dim, virtual_dim, virtual_dim)` → `jnp.zeros` + `flax.linen.param`。Contract 用 `jnp.einsum`。开放边界条件 (e_0 边界向量) 不变。

### 6.4 bitspack

`pack_int`/`unpack_int` 的 `torch.jit.script` 版本 → 用 `jax.jit` + `jnp.packbits`/`jnp.unpackbits` 重写 (JAX 有原生 bit 操作)。

## 7. 依赖变更

| 当前 (refact 分支) | 迁移后 |
|---|---|
| torch | jax + jaxlib |
| torch.utils.cpp_extension | CMake + nvcc 编译 .so |
| ninja + pybind11 | ctypes + `XLA_FFI_DEFINE_HANDLER_SYMBOL` |
| numpy | jax.numpy (jnp) |
| platformdirs | 保留 (缓存管理) |
| hatchling + hatch-vcs | 保留 (构建系统不变) |
| ruff + ty + pytest | 保留 (开发工具不变) |
| uv | 保留 (依赖管理不变) |

## 8. 实施计划

### Phase 1: CUDA kernel 编译为独立 .so + JAX FFI 注册
- [ ] CMake 构建系统，编译 CUDA kernel → `libqmp_hamiltonian.so`
- [ ] 每个 kernel 定义 `XLA_FFI_DEFINE_HANDLER_SYMBOL` handler
- [ ] `ctypes` 加载 `.so`，逐个 `jax.ffi.register_ffi_target`
- [ ] Python 端 wrap 为 `jax.ffi.ffi_call` callable
- [ ] 验证 JAX 环境下性能无损

### Phase 2: CUDA kernel 算法升级 + 变量重命名
- [ ] `create_mask`/`annihilate_mask`/`flip_mask`/`parity_mask`/`parity_const` 表示层
- [ ] `apply_within`: 添加二级索引二分查找
- [ ] `list_relative`: 替换 Trie 为 cuCollections static_map
- [ ] `find_relative`: 替换 mutex heap 为哈希表 + compact + 阈值加速
- [ ] `diagonal_term`: 不变

### Phase 3: JAX 框架迁移
- [ ] 网络层 (MLP/Transformers/MPS) 用 Flax 重写
- [ ] bitspack 用 JAX 原生操作重写
- [ ] 算法层 (HAAR/VMC/Lanczos) 用 JAX 重写
- [ ] 损失函数用 `jax.jit` + `jax.grad` 重写
- [ ] 模型层 (FCIDUMP/Hubbard/Ising/PySCF/OpenFermion) 适配 JAX
- [ ] 配置系统适配

### Phase 4: 多节点集成
- [ ] shard_map 数据分发 (configs batch 维度, Hamiltonian 复制)
- [ ] 跨节点测试 (单机多卡 → 多机多卡)
- [ ] 性能基准对比 (vs PyTorch baseline)

## 9. 参考资料

- JAX FFI: https://jax.readthedocs.io/en/latest/jax/extend/ffi.html
- JAX shard_map: https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html
- cuCollections: https://github.com/NVIDIA/cuCollections
- CUB: https://nvidia.github.io/cccl/cub/
- NetKet (JAX VMC reference): https://github.com/netket/netket
- FermiNet (JAX NN-VMC reference): https://github.com/google-deepmind/ferminet
