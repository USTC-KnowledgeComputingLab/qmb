# Spec: Fermi Hamiltonian 子系统实现

**日期**: 2026-06-17
**状态**: approved

## 1. 目标

实现 `src/qmp/hamiltonian/fermi_hamiltonian/` 子包，包含：

- **位掩码预处理** (`prepare`): 纯 Python 将费米子哈密顿量字典转为 (create_mask, annihilate_mask, flip_mask, parity_mask, parity_const, coef) 表示
- **四个 CUDA kernel**: `compute_diagonal_within_subspace`, `apply_within_subspace`, `find_all_relative_configs`, `find_topk_relative_configs` — 通过 JAX FFI 接入
- **纯 JAX fallback**: CPU/CI 环境下无 GPU 可用时，用 `jax.jit` 实现相同功能
- **JIT 编译与缓存**: CUDA kernel 按需编译 (.cu → .so)，缓存于 `~/.cache/qmp/`
- **pytest 测试**: prepare 正确性 + fallback 端到端 + CUDA 回归

## 2. 文件结构

```
src/qmp/hamiltonian/fermi_hamiltonian/
├── AGENTS.md                    # 子系统设计文档
├── __init__.py                  # re-export FermiHamiltonian
├── _hamiltonian.py              # FermiHamiltonian 类 + FFI 注册 + CUDA/fallback 路由
├── _hamiltonian_prepare.py      # 纯 Python 位掩码预处理 → JAX arrays
├── _hamiltonian_jax.py          # 纯 JAX fallback 实现 (四个操作, @jax.jit)
├── _hamiltonian_cuda.cu         # CUDA kernel (四个操作, 模板化 n_qubytes/max_op_number)
└── _hamiltonian_cuda_loader.py  # CUDA JIT 编译 + 缓存 (nvcc + platformdirs)

tests/
├── __init__.py
└── unit/
    ├── __init__.py
    └── hamiltonian/
        ├── __init__.py
        ├── test_prepare.py       # prepare 正确性测试
        ├── test_fallback.py      # 纯 JAX fallback 四个操作端到端
        └── test_cuda.py          # CUDA vs fallback 回归 (pytest.mark.cuda)
```

## 3. 模块设计

### 命名约定

| 操作 | Python 方法 (FermiHamiltonian) | XLA FFI target | C++ handler | 语义 |
|------|------------|----------------|-------------|------|
| 对角元 | `compute_diagonal_within_subspace` | `qmp_compute_diagonal_within_subspace` | `ComputeDiagonalWithinSubspace` | H[i,i] |
| H·psi 投影 | `apply_within_subspace` | `qmp_apply_within_subspace` | `ApplyWithinSubspace` | H|ψ⟩ 投影到目标子空间 |
| 全部枚举 | `find_all_relative_configs` | `qmp_find_all_relative_configs` | `FindAllRelativeConfigs` | 所有新构型 + 去重 |
| Top-K 选择 | `find_topk_relative_configs` | `qmp_find_topk_relative_configs` | `FindTopKRelativeConfigs` | 最重要的 K 个新构型 |

### 3.1 `_hamiltonian_prepare.py` — 位掩码预处理

**输入**: `hamiltonian: dict[tuple[tuple[int,int],...], complex], n_qubits: int`

**输出**: 6 个 JAX arrays (on CPU):
| 名称 | shape | dtype | 含义 |
|------|-------|-------|------|
| `create_mask` | [T, Q] | uint8 | 须为 0 的位 (产生算符目标) |
| `annihilate_mask` | [T, Q] | uint8 | 须为 1 的位 (湮灭算符目标) |
| `flip_mask` | [T, Q] | uint8 | 翻转掩码 (作用后 XOR) |
| `parity_mask` | [T, Q] | uint8 | JW 奇偶性掩码 |
| `parity_const` | [T] | uint8 | 固定奇偶性 bit (0/1) |
| `coef` | [T, 2] | float64 | 复数系数 (real, imag) |

T = 有效 term 数量, Q = ceil(n_qubits/8)。

**算法**（伪码级别）:

```python
def prepare(operators, n_qubits):
    """
    operators: list[(site, kind)] in WRITING order. kind=1=create, kind=0=annihilate.
    Returns: (create_mask, annihilate_mask, flip_mask, parity_mask, parity_const)
    """
    ops = list(reversed(operators))  # 最右边的算符先作用

    known   = [False] * n_qubits  # qubit i 的初始构型是否已被约束
    initial = [0]     * n_qubits  # 被约束的初始值 (仅 known[i] 时有效)
    flip    = 0                    # 已处理算符的累计翻转位掩码
    p_const = 0                    # JW 相位: 全部确定性贡献 (已知 qubit 的 initial⊕flip)
    p_mask  = 0                    # JW 相位: 运行时构型依赖部分 (仅未知 qubit)

    for s, c in ops:               # c=1(产生) or 0(湮灭)
        # ---- a) 约束推导 ----
        flip_s = (flip >> s) & 1   # qubit s 被之前的算符翻转了奇数次?
        target = 1 - c             # 产生要求中间态=0, 湮灭要求中间态=1
        if known[s]:
            # 之前已有算符约束了 qubit s, 验证中间态是否满足
            if (initial[s] ^ flip_s) != target:
                return ZERO
        else:
            # 首次触及 qubit s, flip_s 必为 0 (无更早的算符翻转它)
            # 中间态 = 初始值 = target
            initial[s] = target
            known[s] = True

        # ---- b) JW 相位贡献 ----
        lo = (1 << s) - 1
        # 对已知 qubit (known[j]=true)：initial[j] XOR flip[j] 已完全确定 → 直接计入 p_const
        # 对未知 qubit (known[j]=false)：flip[j]=0 (不变量)，贡献 = initial[j] → 计入 p_mask
        # 将两者分离：已知的全部放入 p_const，未知的全部放入 p_mask
        known_mask = sum((1 << i) for i in range(n_qubits) if known[i])
        unknown_mask = lo & ~known_mask
        p_const ^= sum((initial[i] ^ ((flip >> i) & 1)) for i in range(s) if known[i]) & 1
        p_mask  ^= unknown_mask

        # ---- c) 翻转 (在相位计算之后) ----
        flip ^= (1 << s)

    # ---- 组装输出 ----
    create_mask = 0
    annihilate_mask = 0
    for i in range(n_qubits):
        if known[i] and initial[i] == 0: create_mask      |= (1 << i)
        if known[i] and initial[i] == 1: annihilate_mask  |= (1 << i)

    return (create_mask, annihilate_mask, flip, p_mask, p_const)
```

**不变量**: `known[s]=false` ⇒ `flip[s]=0`（首次触及必定无历史翻转）。因此未知 qubit 在 p_mask 中贡献的运行时 `config[i]` 恰好等于中间态 `initial[i]`（无翻转修正）。已知 qubit 的 `initial[i] XOR flip[i]` 已完全确定，直接计入 `p_const`。

**边缘情形**:
| 项 | 作用序 | 过程 | 结果 |
|----|--------|------|------|
| `c_i c_i` | c_i, c_i | 首次: init[i]=1, flip={i}; 再次: init[i]=1^flip_s(1)=0≠target(1) → ZERO | ✓ |
| `c_i^dag c_i` | c_i, c_i^dag | 首次: init[i]=1, flip={i}; 再次: init[i]=1^flip_s(1)=0=target(产生要求0) ✓ | flip_mask=0, annihilate_mask={i} |
| `c_i c_i^dag` | c_i^dag, c_i | 首次: init[i]=0, flip={i}; 再次: init[i]=0^flip_s(1)=1=target(湮灭要求1) ✓ | flip_mask=0, create_mask={i} |
| `kind=2` | — | 跳过 | — |

### 3.2 `_hamiltonian_cuda.cu` — CUDA kernel

**模板参数** (编译期宏):
- `N_QUBYTES`: ceil(n_qubits/8)，同时作为静态分发键
- `MAX_OP_NUMBER`: 每 term 最大算符数 (二体相互作用 = 4)

**静态分发**: `N_QUBYTES` 决定内核使用哪种位运算路径。通过 `if constexpr (N_QUBYTES <= 8)` 在编译期选择:
- **n_qubits ≤ 64** (N_QUBYTES ≤ 8): 所有掩码可装载为单个 `uint64_t` 寄存器。可作用性检查 `(config & create_mask) == 0` 是单条 AND+JZ 指令，JW 奇偶性是单条 POPCNT 指令。零循环开销。
- **n_qubits > 64** (N_QUBYTES > 8): 掩码以 `std::array<uint8_t, N_QUBYTES>` 存储，逐字节循环。循环边界是编译期常量，编译器可完全展开。

#### 3.2.1 `compute_diagonal_within_subspace` — 对角元

计算 H 在每个 config 上的对角元期望值。只有 `flip_mask[t] == 0` 的 term（不改变构型）才对对角有贡献。

```
for each term_t:
    for each config_i:
        if not is_applicable(config_i, create_mask[t], annihilate_mask[t]): continue
        if flip_mask[t] != 0: continue   # 对角条件: 无净翻转

        parity = parity_const[t] XOR popcount(parity_mask[t] & config_i) & 1
        sign = -1.0 if parity else 1.0

        atomicAdd(psi[i,0], sign * coef[t,0])
        atomicAdd(psi[i,1], sign * coef[t,1])
```

**并行度**: grid-stride loop，grid 大小按 SM 数量 × 最大 occupancy 设定（~10^5 blocks 级别，而非 T×B = 10^12）。每个 block 内循环处理多个 term × config 对。

**预过滤**: Python 层预处理阶段将 term 按 `flip_mask[t] == 0` 分为对角 term 和非对角 term。`compute_diagonal_within_subspace` 仅遍历对角 term 子集（diagonal terms 通常只占 1-10% 总 term 数），减少 90-99% 无用线程。filtered_terms 作为输入传入 kernel。

**块级归约**: 每个 block 的多个线程可能对同一个 `psi[i]` 累加贡献。先用 shared memory 做 intra-block 归约，最后每个 block 只发一次 `atomicAdd`。这在 diagonal term 密集时（多 term 对应同一 config）显著减少 L2 原子争用。

#### 3.2.2 `apply_within_subspace` — 稀疏 H·psi 投影

计算 `ψ_j = H · ψ_i` 投影到 `configs_j` 子空间。支持 forward/backward 双向。

```
预处理: 对 dst_configs 构建 cuco::static_map (key=config, value=index_in_original_order)
         10^7 entries → ~500 MB (60% load factor)

for each term_t:
    for each src_i:                              # forward: src=configs_i; backward: src=configs_j
        if not is_applicable(src_i, create_mask[t], annihilate_mask[t]): continue
        new_config = src_i XOR flip_mask[t]

        idx = hash_table.lookup(new_config)       # O(1) expected, ~2-3 probes
        if idx < 0: continue

        parity = parity_const[t] XOR popcount(parity_mask[t] & src_i) & 1
        sign = -1.0 if parity else 1.0
        contribution = sign * complex_mul(coef[t], psi_src[i])
        atomicAdd(psi_j[idx,0], contribution.real)
        atomicAdd(psi_j[idx,1], contribution.imag)
```

**方向选择**: forward 遍历 T × B_i，backward 遍历 T × B_j。选较小侧最小化总线程数。forward 时 `psi_i` 为输入波函数，backward 时 `psi_i` 为 `configs_j` 上的波函数（H^dag 作用于它，投影回 `configs_i`）。输出形状始终为 `[B_j, 2]`。

- **`__ldg()` 读取掩码和数据**: 所有只读输入通过 `__ldg()` 走 read-only cache，减少 L1 压力。
- **哈希表跨调用缓存**: 若 `configs_j` 在多轮迭代中不变（Lanczos 内循环常见），复用哈希表省 50-200ms 构建时间。以 `configs_j` 数据指针的 hash 作为 key 判断是否需要重建。
- **哈希函数**: 使用 xxHash64。config bytes 非随机（受占据数、自旋守恒约束），MurmurHash3 在结构化数据上可能产生聚集。

#### 3.2.3 `find_all_relative_configs` — 全部枚举 + 去重

枚举 H 作用产生的所有不重复新 config，累加各路径贡献的振幅。

```
初始化 cuCollections cuco::static_map (capacity = estimated_distinct / 0.6)

for each term_t:
    for each config_i:
        if not is_applicable(config_i, create_mask[t], annihilate_mask[t]): continue
        new_config = config_i XOR flip_mask[t]

        # 排除已知构型: Bloom 预筛查 → 二分查找确认
        if bloom_maybe_present(exclude_set, new_config):
            if binary_search(exclude_configs, new_config) >= 0: continue

        # 计算贡献
        parity = parity_const[t] XOR popcount(parity_mask[t] & config_i) & 1
        sign = -1.0 if parity else 1.0
        contribution = sign * complex_mul(coef[t], psi_i[i])

        # 哈希表: 找到则 atomicAdd, 未找到则 CAS 声明 slot
        slot = hash_table.find(new_config)
        if slot: atomicAdd(slot, contribution)
        else: hash_table.insert_cas(new_config, contribution)

return hash_table.collect_nonempty()  # 线性扫描
```

**容量**: 10^7 distinct → ~600 MB; 10^8 → ~6 GB。预分配，零运行时分配。

**溢出保护**: 插入时 probe 超阈值（如 10×log₂(capacity)）→ 标记溢出 → kernel 返回错误码 → Python 层用更大 capacity 重试。不可静默丢弃 config。

**排除集哈希表**: 用第二个 `cuco::static_map` 代替 Bloom+二分查找。10^8 exclude entries → 额外 ~6 GB HBM，总计 ~12 GB——在 80 GB H100 上可行，且避免了 Bloom 假阳性带来的 O(log N) 二分查找回退。

#### 3.2.4 `find_topk_relative_configs` — Top-K 选择

选出最重要的 K 个新 config。重要性度量 = `max |H_ij * psi_i|²`（单路径最强连接的权重）。

```
初始化 哈希表 (capacity = 2K, ~680 MB, 驻留 HBM)
global_min_weight = 0.0
chunk_size = max_terms_per_launch  # 按 GPU 能力动态确定，通常 ~1000

for chunk_start = 0; chunk_start < T; chunk_start += chunk_size:
    # ── CUDA kernel: 处理 chunk 内的 terms × all configs ──
    for t in terms[chunk_start : chunk_start + chunk_size]:
        for each config_i:
            if not is_applicable(config_i, create_mask[t], annihilate_mask[t]): continue
            weight = |coef[t] * psi_i[i]|²
            if weight <= global_min_weight: continue   # 阈值快速拒绝

            new_config = config_i XOR flip_mask[t]
            if in_exclude_set(new_config): continue

            slot = hash_table.find(new_config)
            if slot: atomicMax(slot.weight, weight)
            else: hash_table.insert_cas(new_config, weight)
    # ── kernel 退出 (隐式全局屏障) ──

    # ── compact (独立 kernel 或 host 端) ──
    compact():
        entries = hash_table.collect_nonempty()
        CUB::radix_sort_descending(entries, key=weight)
        entries = unique_top_k(entries, K)
        hash_table.rebuild(entries)             # 重建仅含 top K
        global_min_weight = entries[-1].weight  # 更新阈值

# 最终 compact
final_compact(): CUB radix sort desc → unique top K → return top_K_configs
```

**设计决策**: 紧凑哈希表 + kernel 间 compact。compact 不在 kernel 内部——每个 chunk 的 kernel 退出本身就是隐式全局屏障，此时安全执行 sort→rebuild→update threshold→启动下一个 chunk。

**为何 `max` 而非 `sum`**: 量子化学权重分布近似 power-law，`max` 和 `sum` 强相关（Spearman ρ ≈ 0.75-0.95）。且 `atomicMax` 单调整、确定性、无浮点精度问题，适合流式聚合。

**容量**: 2K entries（K×2 overprovisioning）→ hash table ~680 MB（vs 全量 5.7 GB）。阈值单调上升，前几轮 compact 后 ~99.9% 条目被快速拒绝。

**chunk 大小**: 按 GPU max resident blocks 动态确定，确保每轮 kernel 足额利用 GPU，但不受限于最大并行度（超出部分由 grid-stride loop 处理）。~100 轮 compact 总开销约 50ms。

### 3.3 `_hamiltonian_cuda_loader.py` — JIT 编译与缓存

```python
def load_cuda_module(n_qubytes: int, max_op_number: int) -> ctypes.CDLL:
    key = f"qmp_hamiltonian_{n_qubytes}_{max_op_number}"
    cache_dir = platformdirs.user_cache_path("qmp", "kclab") / key
    so_path = cache_dir / "lib.so"

    if not so_path.exists():
        subprocess.run([
            "nvcc", "-shared", "-Xcompiler", "-fPIC",
            f"-I{jaxlib.get_include_dir()}",
            f"-DN_QUBYTES={n_qubytes}",
            f"-DMAX_OP_NUMBER={max_op_number}",
            "-std=c++20", "-O3", "--use_fast_math",
            "-arch=native",
            "-o", str(so_path),
            str(SOURCE_DIR / "_hamiltonian_cuda.cu"),
        ], check=True)

    lib = ctypes.cdll.LoadLibrary(str(so_path))
    return lib
```

### 3.4 `_hamiltonian_jax.py` — 纯 JAX fallback

四个 `@jax.jit` 函数，算法与 3.2.1-3.2.4 完全一致，区别仅在循环结构: CUDA 用 2D grid (`term × batch` 维度并行 + `atomicAdd`)，JAX 用 `vmap` + `lax.scan` 或嵌套 `fori_loop`。两套实现共享相同的位掩码输入和输出语义。

### 3.5 `_hamiltonian.py` — FermiHamiltonian 类 + FFI 路由

```python
class FermiHamiltonian:
    def __init__(self, hamiltonian_dict, *, n_qubits, devices):
        arrays = prepare(hamiltonian_dict, n_qubits)
        self._create_mask, self._annihilate_mask, self._flip_mask, ...
        self._device = ...
        # 按 |coef| 降序排列
        # 尝试加载 CUDA kernel
        self._use_cuda = _try_enable_cuda(...)

    def compute_diagonal_within_subspace(self, configs):
        ...
    def apply_within_subspace(self, configs_i, psi_i, configs_j, *, direction=0):
        ...
    def find_all_relative_configs(self, configs_i, psi_i, configs_exclude, *, hash_capacity):
        ...
    def find_topk_relative_configs(self, configs_i, psi_i, count_selected, configs_exclude):
        ...
```

## 4. 依赖

- **运行时**: `jax`, `jaxlib` (>=0.5.0), `platformdirs`
- **编译 CUDA**: `nvcc` (用户机器上), `jaxlib` 的 XLA headers
- **无**: numpy, torch, pybind11, ninja

## 5. 测试

### 5.1 `tests/unit/hamiltonian/test_prepare.py`
- `test_create_mask_h2`: H₂ 哈密顿量 verify create_mask
- `test_annihilate_mask_hubbard_2x1`: 2-site Hubbard verify annihilate_mask
- `test_parity_mask_jw`: JW 符号正确性 (已知手工计算)
- `test_zero_term_skip`: 恒为零的 term 被正确跳过
- `test_coef_preserved`: 系数保持

### 5.2 `tests/unit/hamiltonian/test_fallback.py`
- 小型合成哈密顿量 (4 qubits, ~10 terms), 10-20 configs
- `test_diagonal_term_exact`: 手工计算结果对比
- `test_apply_within_subspace_forward_backward_consistency`: H 和 H^dag 结果关系
- `test_find_all_relative_configs_dedup`: 去重和振幅累加
- `test_find_topk_relative_configs_topk`: Top-K 排序

### 5.3 `tests/unit/hamiltonian/test_cuda.py`
- `@pytest.mark.cuda` — 仅 GPU 环境运行
- 与 test_fallback 相同输入，assert CUDA 输出 ≈ JAX fallback 输出 (`allclose(rtol=1e-12)`)
- **非 bit-exact**: GPU 的 `atomicAdd` 顺序非确定，浮点加法非结合律导致末位差异，不能用 `==` 比较
- `test_hash_table_overflow_retry`: 人为给定过小 capacity，验证重试逻辑

## 6. 非目标

- 多节点 shard_map 集成: 不在此 spec 中 (基础设施已预留)

## 7. 已知风险

| 风险 | 等级 | 缓解 |
|------|------|------|
| 哈希表溢出导致 kernel hang (`cuco::static_map` open addressing 无限 probe) | **高** | 插入失败检测 (probe 超阈值 10×log₂(capacity)) → 错误码 → Python 层 retry 更大 capacity |
| `compute_diagonal` 对角 term 仅占 1-10%，grid-stride loop 避免 10^12 线程启动 | **高** | 预过滤对角 term 子集 + grid-stride loop (~10^5 blocks)，不用 T×B 网格 |
| 结构化 config bytes 导致 MurmurHash3 聚集 | 中 | 所有哈希表统一使用 xxHash64 |
| L2 cache thrashing（configs 流驱逐 hash table 条目） | 中 | configs 用 `cudaAccessPropertyStreaming` 标记，`__ldg()` 走 read-only cache |
| 重复 alloc/free 500 MB 哈希表导致 HBM 碎片化 | 中 | `cudaMemPool` 预分配；apply_within 哈希表跨调用缓存 |
| find_topk 多 kernel 分段中 compact 开销（~100 轮 × 0.5ms） | 低 | 总开销 ~50ms，相对 10+ 分钟总体计算忽略不计 |
| 纯 JAX fallback 与 CUDA 非 bit-exact (`atomicAdd` 顺序非确定) | 低 | 测试用 `allclose(rtol=1e-12)` |
| `exclude_configs` 未排序传入 CUDA | 低 | Python 层预处理排序，或用第二哈希表替代 Bloom+二分查找
