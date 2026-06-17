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

- **构建后端**: hatchling + hatch-vcs (setuptools → hatchling)
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
│   ├── hamiltonian/        # CUDA C++ kernel + Python thin wrapper
│   │   ├── __init__.py
│   │   ├── AGENTS.md
│   │   ├── _hamiltonian.py          # Python thin wrapper
│   │   ├── _hamiltonian.cpp         # prepare + TORCH_LIBRARY_FRAGMENT
│   │   ├── _hamiltonian_cpu.cpp     # CPU backend (TORCH_LIBRARY_IMPL)
│   │   ├── _hamiltonian_cuda.cu     # CUDA backend (TORCH_LIBRARY_IMPL)
│   │   ├── _spin_separated_hamiltonian.py
│   │   ├── _spin_separated_hamiltonian.cpp
│   │   ├── _spin_separated_hamiltonian_cpu.cpp
│   │   └── _spin_separated_hamiltonian_cuda.cu
│   ├── networks/            # MLP / Transformers / MPS (→ Flax)
│   ├── algorithms/          # HAAR / VMC / Lanczos (→ 纯 JAX)
│   ├── models/              # FCIDUMP / Hubbard / Ising / PySCF / OpenFermion
│   ├── plugins/             # 第三方框架接口
│   └── utility/             # bitspack, losses, context, optimizer
├── tests/
├── docs/superpowers/
│   ├── specs/               # 设计 spec
│   └── plans/               # 实施 plan
└── old/                     # main 分支参考代码
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

每个 Hamiltonian 操作通过 `jax.extend.ffi` 注册为 XLA custom call:

```python
import jax
from jax.extend import ffi

# Python 端注册
ffi.register_ffi_target("qmp_apply_within", cpp_capsule, platform="CUDA")

@ffi.ffi_call("qmp_apply_within", result_shape_dtypes, vmap_method="sequential")
def apply_within(configs_i, psi_i, configs_j, site, kind, coef):
    ...

# 与 shard_map 配合使用
@jax.shard_map(mesh=mesh, in_specs=P('configs'), out_specs=P('configs'))
def distributed_apply_within(configs_shard, psi_shard, ...):
    return apply_within(configs_shard, psi_shard, ...)
```

### 3.2 迁移策略

1. **网络层**: MLP/Transformers/MPS 算法逻辑不变，用 Flax 重写，`torch.jit.script` → `jax.jit`
2. **算法层**: HAAR/VMC/Lanczos 用纯 JAX 重写，`jax.grad` 替代手写 closure
3. **Hamiltonian 层**: CUDA kernel 通过 JAX FFI 接入，binding 层从 `torch.utils.cpp_extension` 改为 `xla/ffi/api/ffi.h`

## 4. Hamiltonian 层详细设计

### 4.0 现有架构 (refact 分支已确立的四层模式)

```
每个操作遵循相同的分层结构:

1. hamiltonian_apply_kernel  (__device__)  — 纯计算核心，施加算符到组态上
2. {operation}_kernel        (__device__)  — 每个 (term, batch) 对的工作单元
3. {operation}_kernel_interface (__global__)  — 遍历所有 term×batch 对
4. {operation}_interface     (host, TORCH_LIBRARY_IMPL)  — PyTorch 集成层
```

**CPU/CUDA 对称性原则**: 两边核心逻辑完全一致，平台差异局限于设备管理、kernel 启动、同步。`CUDAGuard` + `getCurrentCUDAStream` + `cudaDeviceProp` 而非裸 `cudaDeviceSynchronize`。`thrust::sort_by_key` 与 kernel 同流而非回 CPU `std::sort`。

**模板参数命名**: `n_qubytes`, `particle_cut`, `max_op_number`, `forward`，贯穿所有 C++ 签名、模块名、缓存 key。编译期通过 `-DN_QUBYTES=X -DPARTICLE_CUT=Y` 传入。

**编译与绑定**: 当前通过 `torch.utils.cpp_extension.load` JIT 编译，缓存于 `~/.cache/qmp/`:

```
_hamiltonian.cpp       → 声明模块 (prepare 函数 + TORCH_LIBRARY_FRAGMENT)
_hamiltonian_cpu.cpp   → CPU 实现 (TORCH_LIBRARY_IMPL(..., CPU, ...))
_hamiltonian_cuda.cu   → CUDA 实现 (TORCH_LIBRARY_IMPL(..., CUDA, ...))
```

**SpinSeparatedHamiltonian** (refact 分支新增): Hamiltonian 的变体，内部将自旋上下分离为 block 排列。prepare 阶段将每个 term 按偶数(up)/奇数(down)分拆为 5 张量输出。支持 `use_lookup_table` 开关将 binary search (O(log N)) 变为查表 (O(1))。

### 4.1 CUDA kernel 迁移: 三阶段

**阶段 1 (当前)**: 保留 Torch C++ extension 体系，DLPack 零拷贝桥接
- 继续用 `torch.utils.cpp_extension.load` 编译 C++/CUDA
- Python 端通过 `torch.utils.dlpack.to_dlpack` / `jax.dlpack.from_dlpack` 做零拷贝 tensor 交换
- 优点: CUDA 代码一行不改，四层架构完全保留，迁移风险低
- 缺点: 每次调用有 DLPack 转换开销（但量级小）

**阶段 2 (中期)**: CUDA kernel 编译为独立共享库，通过 XLA FFI 直接调用
- 保留四层架构中的第 1-3 层 (纯 CUDA 逻辑)，替换第 4 层 (Torch → XLA FFI)
- 用 `nvcc` 编译 `.so`，导出 C ABI 函数
- 用 `jax.extend.ffi` 注册为 XLA custom call target
- 优点: 零转换开销，JIT 可以融合优化
- CUDA 代码几乎不改 (仅函数签名改为接受 raw pointer + strides)

**阶段 3 (远期可选)**: Pallas 重写或全面 XLA 化 — 当前不做此承诺

### 4.2 diagonal_term

**操作**: 对每个 config，累加所有不改变该 config 的哈密顿项系数，得到对角元能量。

**当前实现**: 2D grid (term × batch)，施加项后比较 config 是否不变，`atomicAdd` 累加。

**分析**: Embarrassingly parallel。每个 (term, config) 对完全独立。无去重需求，无排序需求。当前实现已经最优。

**方案**: 保留当前 CUDA kernel (四层架构不变)，仅换 binding 层。

```
FFI 签名:
diagonal_term(
  configs  [B, Q] uint8,
  site     [T, 4] int16,
  kind     [T, 4] uint8,
  coef     [T, 2] f64
) → psi [B, 2] f64 (real + imag)
```

**多卡方案**: configs 按 batch 维度分片，每卡持有完整 Hamiltonian 副本，独立计算，结果天然分片，无需跨卡通信。

### 4.3 apply_within

**操作**: 稀疏矩阵乘向量。输入 configs_i (源空间) + psi_i + configs_j (目标空间)，计算 H·ψ_i 投影到目标空间的结果 ψ_j。

**当前实现** (refact 分支已支持 `forward`/`backward` 双向):
1. `thrust::sort_by_key` 对 configs_j 排序，返回 tuple `(sorted, sort_idx)`
2. 2D grid (dim3{1, maxThreadsPerBlock >> 1}): 每个 (term, config_i) 施加 H 项 → 排序数组中二分查找 → `atomicAdd`

**分析**: configs_j 排序是一次性操作 (O(B_j log B_j))。二分查找每次 ~24 次全局内存随机访问。block 配置 `dim3{1, ~512}` 的原因: term 在 block 间并行，batch 在 block 内并行，site/kind 可广播到全 warp。

**改进**: **二级索引二分查找**
- L1 索引: 每 256 条 config_j 取样 → ~40K 条目，存 `__constant__` 内存
- 先在 L1 索引中二分查找 (~15 次 constant memory 访问，~20 cycles)
- 然后在 256 条目的桶内线性扫描 (1-2 个 cache line)
- 全局内存访问从 ~24 次降到 ~2 次

```
kernel 内部:
  // L1 index 二分查找 (全在 constant memory)
  int bucket = binary_search_l1(l1_index, target_config);
  // 桶内线性扫描
  int idx = linear_scan_bucket(configs_j, bucket * 256, target_config);
```

**方案**: 保留四层架构 + 排序逻辑，在第 2 层 (`apply_within_kernel`) 添加二级索引。

```
FFI 签名:
apply_within(
  configs_i [B_i, Q] uint8,
  psi_i     [B_i, 2] f64,
  configs_j [B_j, Q] uint8,
  site      [T, 4] int16,
  kind      [T, 4] uint8,
  coef      [T, 2] f64
) → psi_j [B_j, 2] f64
```

**多卡方案**: configs_j 按 batch 维度分片，每卡独立计算自己的 psi_j 片段。

### 4.4 list_relative

**操作**: 全部列举 + 去重 + 振幅累加。对每个 (term, config_i) 施加 H 项得到新 config，排除已知 configs，去重并累加相同 config 的振幅。返回所有不重复的新 config 及其累加振幅。

**规模**: 预期 distinct new configs ≈ 10^7-10^8。

**当前实现**: 256叉前缀树 (Trie)，device malloc 分配节点，atomicCAS 锁创建。四层架构: `list_relative_kernel` → `list_relative_kernel_interface` → `list_relative_interface`。

**问题**: Trie 每 distinct entry 占用 ~25KB (N_levels × ~2.5KB/节点)，10^8 entries 需 2.5 TB 显存——不可行。且 device malloc 极慢。

**方案**: **预分配哈希表 (cuCollections `cuco::static_map`)**

```
数据结构:
  static_map<Key=config_bytes, Value=(real+f64, imag+f64)>
  容量: 预估 distinct / 0.6 (60% 负载因子)
  open addressing + linear probing
  预分配，零运行时分配

配置尺寸估算:
  10^7 distinct: ~600 MB (轻松)
  10^8 distinct: ~6 GB   (H100 80GB 内可行)
  10^9 distinct: ~60 GB  (需多卡分区)

插入流程 (per thread):
  1. hamiltonian_apply_kernel → new_config, psi_contribution
  2. 二分查找排除 exclude_configs (或 Bloom filter 预筛查)
  3. 哈希表查找:
     - 找到: atomicAdd(振幅)
     - 未找到: atomicCAS 声明 slot + 写入

收集: 线性扫描哈希表 → 所有非空 slot → (configs, psi)
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
  site             [T, 4] int16,
  kind             [T, 4] uint8,
  coef             [T, 2] f64,
  hash_capacity    int (预估 distinct / 0.6)
) → (new_configs [N, Q] uint8, psi_j [N, 2] f64)
```

### 4.5 find_relative

**操作**: 流式 Top-K 选择。对每个 (term, config_i) 施加 H 项得到候选 (new_config, weight)，排除已知 configs，选出最重要的 K 个不重复新 config。K ≈ 100K (十万量级)。

与 `list_relative` 的区别: 不返回所有新 config，只选 Top-K；重要性度量是 `|H_ij * psi_i|²` (per-path contribution squared)，而非累加振幅。

**当前实现**: 2D grid → 并发最小堆 (per-node mutex + nanosleep backoff)，四层架构同其他操作。

**方案**: **全局哈希表 + 阈值加速 + 周期性 compact**

综合调研后选定的方案: 用哈希表作为"软 Top-K"累加器，定期 compact 到精确 Top-K。

```
数据结构:
  ┌──────────────────────────────────────────────┐
  │ 哈希表 (Global Memory, L2 cached)             │
  │   容量: 2K = 200K slots                       │
  │   (× ~48B/slot ≈ 9.6 MB, fits in L2)         │
  │   负载因子上限: 0.6                           │
  │   open addressing with linear probing         │
  │                                               │
  │   global_min_weight: float64 (阈值)            │
  └──────────────────────────────────────────────┘

流式插入 (per thread):
  1. if weight ≤ global_min_weight → REJECT (~2 cycles)
     → 99.9%+ 条目在此被过滤

  2. 慢路径 (weight > threshold):
     a. 二分查找排除 exclude_configs
     b. 哈希探测 → 找到: atomicMax(weight) ; 未找到: atomicCAS slot
     c. 每个成功的慢路径操作: ~100-200 cycles (L2 resident)

  3. 周期性 compact (当 load_factor > 0.6 或间隔触发):
     ▸ CUB DeviceRadixSort 按 weight 降序排序所有条目
     ▸ 取前 K 个唯一 config
     ▸ 重建哈希表 (仅用 top K)
     ▸ 更新 global_min_weight
     ▸ compact 成本: ~0.3ms/次 (200K 条目排序)
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
4. 实现复杂度: 哈希表主要用现有库；堆需要手写 warp-level 协作逻辑

```
FFI 签名:
find_relative(
  configs_i        [B_i, Q] uint8,
  psi_i            [B_i, 2] f64,
  count_selected   int (K),
  configs_exclude  [E, Q] uint8,
  site             [T, 4] int16,
  kind             [T, 4] uint8,
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
| torch.utils.cpp_extension | jax.extend.ffi (XLA FFI, 阶段2+) |
| ninja + pybind11 | cmake / nvcc 直接编译 .so (阶段2+) |
| numpy | jax.numpy (jnp) |
| platformdirs | 保留 (缓存管理) |
| hatchling + hatch-vcs | 保留 (构建系统不变) |
| ruff + ty + pytest | 保留 (开发工具不变) |
| uv | 保留 (依赖管理不变) |

## 8. 实施计划

### Phase 1: FFI 绑定层 (DLPack 零拷贝)
- [ ] 将每个 CUDA kernel 的 Python wrapper 改为 DLPack 零拷贝调用
- [ ] 验证 PyTorch 环境下性能无损
- [ ] 保留四层 C++ 架构不变

### Phase 2: CUDA kernel 算法升级
- [ ] `apply_within`: 添加二级索引二分查找
- [ ] `list_relative`: 替换 Trie 为 cuCollections static_map
- [ ] `find_relative`: 替换 mutex heap 为哈希表 + compact + 阈值加速
- [ ] `diagonal_term`: 不变
- [ ] `SpinSeparatedHamiltonian`: 对应升级各操作

### Phase 3: JAX 框架迁移
- [ ] 网络层 (MLP/Transformers/MPS) 用 Flax 重写
- [ ] bitspack 用 JAX 原生操作重写
- [ ] 算法层 (HAAR/VMC/Lanczos) 用 JAX 重写
- [ ] 损失函数用 `jax.jit` + `jax.grad` 重写
- [ ] 模型层 (FCIDUMP/Hubbard/Ising/PySCF/OpenFermion) 适配 JAX
- [ ] 配置系统适配

### Phase 4: 多节点集成
- [ ] CUDA kernel 编译为独立 .so，注册到 XLA FFI
- [ ] shard_map 数据分发
- [ ] 跨节点测试 (单机多卡 → 多机多卡)
- [ ] 性能基准对比 (vs PyTorch baseline)

## 9. 参考资料

- JAX FFI: https://jax.readthedocs.io/en/latest/jax/extend/ffi.html
- JAX shard_map: https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html
- cuCollections: https://github.com/NVIDIA/cuCollections
- CUB: https://nvidia.github.io/cccl/cub/
- NetKet (JAX VMC reference): https://github.com/netket/netket
- FermiNet (JAX NN-VMC reference): https://github.com/google-deepmind/ferminet
- qmp-kit refact branch: `refact` (本仓库另一个 worktree)
