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
├── _hamiltonian_cuda.cu         # CUDA kernel (四个操作, 模板化 n_qubytes)
└── _hamiltonian_cuda_loader.py  # CUDA JIT 编译 + 缓存 (nvcc + platformdirs)

tests/
├── __init__.py
└── unit/
    ├── __init__.py
    └── hamiltonian/
        └── fermi_hamiltonian/
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

> **伪代码命名**: 本文使用 `B`/`T`/`Q`/`K` 等单字母大写符号仅作伪代码示意。实际代码中必须使用描述性名称: `batch_size`/`term_count`/`n_qubytes`/`count_selected`。见 AGENTS.md 命名约定。

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
- `N_QUBYTES` (macro): ceil(n_qubits/8)，同时作为静态分发键

**静态分发**: 通过 `if constexpr (n_qubytes <= 8)` 在编译期选择:
- **n_qubits ≤ 64** (n_qubytes ≤ 8): 所有掩码可装载为单个 `uint64_t` 寄存器。可作用性检查 `(config & create_mask) == 0` 是单条 AND+JZ 指令，JW 奇偶性是单条 POPCNT 指令。零循环开销。
- **n_qubits > 64** (n_qubytes > 8): 掩码以 `uint8_t[n_qubytes]` 存储，逐字节循环。循环边界是编译期常量，编译器可完全展开。

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

**预过滤** (planned, not yet in active path): Python 层预处理阶段将 term 按 `flip_mask[t] == 0` 分为对角 term 和非对角 term。`compute_diagonal_within_subspace` 可仅遍历对角 term 子集（diagonal terms 通常只占 1-10% 总 term 数），减少 90-99% 无用线程。当前实现中 `_diag_idx` 已计算并写入日志，但 kernel 仍遍历全部 terms 并在运行时检查 `flip_mask[t] == 0`。

**块级归约**: 每个 block 的多个线程可能对同一个 `psi[i]` 累加贡献。先用 shared memory 做 intra-block 归约，最后每个 block 只发一次 `atomicAdd`。这在 diagonal term 密集时（多 term 对应同一 config）显著减少 L2 原子争用。

#### 3.2.2 `apply_within_subspace` — 稀疏 H·psi 投影

计算 `ψ_j = H · ψ_i` 投影到 `configs_j` 子空间。支持 forward/backward 双向。

```
预处理: 对 dst_configs 构建自定义线性探测哈希表 (apply_hash_slot, key=config bytes, value=index)
         哈希: wyhash64, load factor 60%

for each term_t:
    for each src_i:                              # forward: src=configs_i; backward: src=configs_j
        # backward 时需检查 config_i (= config_j XOR flip[t]) 是否满足约束
        check_config = src_i XOR flip_mask[t] if direction == 1 else src_i
        if not is_applicable(check_config, create_mask[t], annihilate_mask[t]): continue
        new_config = src_i XOR flip_mask[t]

        idx = apply_hash_lookup(new_config)       # forward: lookup in configs_j; backward: lookup in configs_i
        if idx < 0: continue

        parity = parity_const[t] XOR popcount(parity_mask[t] & check_config) & 1
        sign = -1.0 if parity else 1.0
        cf = (coef[t,0], -coef[t,1]) if direction == 1 else (coef[t,0], coef[t,1])  # conj for H^†
        contribution = sign * complex_mul(cf, psi_src[i])
        atomicAdd(psi_j[idx,0], contribution.real)
        atomicAdd(psi_j[idx,1], contribution.imag)
```

**方向选择**: forward 遍历 T × B_i，backward 遍历 T × B_j。选较小侧最小化总线程数。forward 时 `psi_i` 为输入波函数，输出形状 `[B_j,2]`。backward 时 `psi_i` 为 `configs_j` 上的波函数，H^† 作用于它，投影回 `configs_i`，输出形状 `[B_i,2]`。

- **`__ldg()` 读取掩码和数据**: 所有只读输入通过 `__ldg()` 走 read-only cache，减少 L1 压力。
- **哈希表跨调用缓存** (planned, placeholder only): 若 `configs_j` 在多轮迭代中不变（Lanczos 内循环常见），复用哈希表省 50-200ms 构建时间。`FermiHamiltonian` 中已预留 `_apply_hash_cache` 字段，但 CUDA kernel 当前每次调用均重新构建哈希表。
- **哈希函数**: 使用 wyhash。config bytes 非随机（受占据数、自旋守恒约束），wyhash 使用 multiply-and-xor 链 (wymum) 对结构化 occupation-number 字节有良好扩散。

#### 3.2.3 `find_all_relative_configs` — 全部枚举 + 去重

枚举 H 作用产生的所有不重复新 config，累加各路径贡献的振幅。

```
初始化自定义 CAS 哈希表 (findall_slot, capacity = estimated_distinct / 0.6)
哈希: wyhash64, linear probing, probe 上限 100

for each term_t:
    for each config_i:
        if not is_applicable(config_i, create_mask[t], annihilate_mask[t]): continue
        new_config = config_i XOR flip_mask[t]

        # 排除已知构型: 线性扫描 configs_exclude
        if linear_scan_match(exclude_configs, new_config) >= 0: continue

        # 计算贡献
        parity = parity_const[t] XOR popcount(parity_mask[t] & config_i) & 1
        sign = -1.0 if parity else 1.0
        contribution = sign * complex_mul(coef[t], psi_i[i])

        # 哈希表: 找到则 atomicAdd, 未找到则 CAS 声明 slot
        slot = hash_table.find(new_config)
        if slot: atomicAdd(slot, contribution)
        else: hash_table.insert_cas(new_config, contribution)

# kernel 退出后 → host 端线性扫描收集非空 slots
```

**容量**: 10^7 distinct → ~600 MB; 10^8 → ~6 GB。预分配，零运行时分配。

**溢出保护**: 插入时 probe 超硬上限 100 → 标记 overflow → 不可静默丢弃 config。Python 层尚未读取 overflow 标志以 retry。

**排除集**: 对 `configs_exclude` 进行线性扫描排除已知构型（terms × configs 数量级不高时可行）。规划中可用第二个 `cuco::static_map` 替代线性扫描以支持更大排除集。

#### 3.2.4 `find_topk_relative_configs` — Top-K 选择

选出最重要的 K 个新 config。

```
初始化 哈希表 (capacity = 2K)

# 单次 kernel 遍历全部 T × B 对
for each term_t:
    for each config_i:
        if not is_applicable(config_i, create_mask[t], annihilate_mask[t]): continue
        weight = |coef[t] * psi_i[i]|²

        new_config = config_i XOR flip_mask[t]
        if in_exclude_set(new_config): continue

        slot = hash_table.find(new_config, weight)
        if slot: atomicMax(slot.weight, weight)
        else: hash_table.insert_cas(new_config, weight)

# kernel 退出后 → Python/JAX 层 argsort → top K
```

**实现**: 当前为单次 kernel 启动。规划中的 between-kernel compaction（chunking + CUB radix sort + rebuild）可在 terms 数量极大时降低内存占用，尚未实现。

**`max` 而非 `sum` 的理由**: (a) `atomicMax` 单调确定，而 `atomicAdd` 浮点非结合导致非确定性；(b) 量子化学中 `|H_ij*c_i|²` 近似 power-law，`max` 与 `sum` 高度相关（Spearman ρ≈0.75-0.95），max 足以选出最重要的 configs；(c) 无需 Gumbel 噪声——其对 K=100K 引入 ~5-15% 排名翻转，且每次额外 60+ cycles 计算成本。

### 3.3 `_hamiltonian_cuda_loader.py` — JIT 编译与缓存

```python
def load_cuda_module(n_qubytes: int) -> ctypes.CDLL:
    key = f"qmp_hamiltonian_{n_qubytes}"
    cache_dir = platformdirs.user_cache_path("qmp", "kclab") / "hamiltonian" / "fermi" / key
    so_path = cache_dir / "lib.so"

    if not so_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        jax_include = str(Path(jaxlib.__file__).parent / "include")
        include_cuco = _THIRD_PARTIES_DIR / "cuco" / "include"
        subprocess.run([
            "nvcc", "-shared", "-Xcompiler", "-fPIC",
            f"-I{jax_include}",
            f"-I{include_cuco}",
            f"-DN_QUBYTES={n_qubytes}",
            "-std=c++20", "-O3", "--use_fast_math",
            "-arch=native",
            "-o", str(so_path),
            str(_SOURCE_DIR / "_hamiltonian_cuda.cu"),
        ], check=True)

    lib = ctypes.cdll.LoadLibrary(str(so_path))
    return lib
```

### 3.4 `_hamiltonian_jax.py` — 纯 JAX fallback

四个函数全部 JIT 兼容，算法与 3.2.1-3.2.4 完全一致:

| 函数 | JIT 方式 | 原因 |
|------|---------|------|
| `compute_diagonal_within_subspace` | `@jax.jit` | 纯算术向量化，无控制流分支 |
| `apply_within_subspace` | `@partial(jax.jit, static_argnums=(9,))` + `jnp.all` vectorized matching | `direction` 用 `static_argnums`；内层查表通过 `jnp.all(dst_c == new_c, axis=1)` 向量化 |
| `find_all_relative_configs` | `@partial(jax.jit, static_argnums=(9,))` + `lax.fori_loop` + `lax.cond` | 哈希表 probe/accumulate 用 `fori_loop` 携带可变状态 + `lax.cond` 分支；`hash_capacity` 编译期固定 |
| `find_topk_relative_configs` | `@partial(jax.jit, static_argnums=(2,))` + `lax.fori_loop` + `lax.cond` | 同上，`count_selected` 编译期固定 |

### 3.5 `_hamiltonian.py` — FermiHamiltonian 类 + FFI 路由

```python
class FermiHamiltonian:
    def __init__(self, hamiltonian_dict, *, n_qubits, devices):
        arrays = prepare(hamiltonian_dict, n_qubits)
        self._create_mask, self._annihilate_mask, self._flip_mask, ...
        self._device = self._parse_device(devices)   # 仅取 devices[0]
        # 按 |coef| 降序排列, 预过滤对角 term
        # 后端由设备平台决定: gpu -> 注册 CUDA FFI (失败则抛错); cpu -> 纯 JAX
        self._use_cuda = self._device.platform == "gpu"
        if self._use_cuda:
            _register_ffi(self._n_qubytes)   # 编译/注册失败会 raise, 不静默 fallback

    def compute_diagonal_within_subspace(self, configs): ...
    def apply_within_subspace(self, configs_i, psi_i, configs_j, *, direction=0): ...
    def find_all_relative_configs(self, configs_i, psi_i, configs_exclude, *, hash_capacity): ...
    def find_topk_relative_configs(self, configs_i, psi_i, count_selected, configs_exclude): ...
```

**后端选择**: 不再用 try/except 静默探测。设备平台是 `gpu` 才尝试 CUDA，失败即报错; `cpu` 直接走 fallback，不触碰 nvcc。

**标量参数**: `direction` / `hash_capacity` / `count_selected` 作为 FFI attribute (`.Attr<int64_t>`) 传递，Python 侧用关键字实参传 Python `int`。

**精度**: kernel 缓冲区为 F64/complex128，调用方须启用 `jax_enable_x64` (测试在 `tests/conftest.py` 全局开启)。

## 4. 依赖

**运行时**: `jax`, `jaxlib` (>=0.5.0), `platformdirs`

**编译 CUDA**: `nvcc` (用户机器上), `jaxlib` 的 XLA headers (`xla/ffi/api/ffi.h`)

**第三方库** (git submodule, 置于 `src/qmp/hamiltonian/third_parties/`):
- `cuCollections`: GPU hash table (`cuco::static_map`), header-only, [github.com/NVIDIA/cuCollections](https://github.com/NVIDIA/cuCollections)
- `wyhash`: 哈希函数, ~30 行 C inline 到 .cu 中, 无需编译

**无**: numpy, torch, pybind11, ninja

## 5. 测试

### 5.1 `tests/unit/hamiltonian/fermi_hamiltonian/test_prepare.py`
- `test_create_mask_h2`: H₂ 哈密顿量 verify create_mask
- `test_annihilate_mask_hubbard_2x1`: 2-site Hubbard verify annihilate_mask
- `test_parity_mask_jw`: JW 符号正确性 (已知手工计算)
- `test_zero_term_skip`: 恒为零的 term 被正确跳过
- `test_coef_preserved`: 系数保持

### 5.2 `tests/unit/hamiltonian/fermi_hamiltonian/test_fallback.py`
- 小型合成哈密顿量 (4 qubits, ~10 terms), 10-20 configs
- `test_diagonal_term_exact`: 手工计算结果对比
- `test_apply_within_subspace_forward_backward_consistency`: H 和 H^dag 结果关系
- `test_find_all_relative_configs_dedup`: 去重和振幅累加
- `test_find_topk_relative_configs_topk`: Top-K 排序

### 5.3 `tests/unit/hamiltonian/fermi_hamiltonian/test_cuda.py`
- `@pytest.mark.cuda` — 仅 GPU 环境运行
- 与 test_fallback 相同输入，assert CUDA 输出 ≈ JAX fallback 输出 (`allclose(rtol=1e-12)`)
- **非 bit-exact**: GPU 的 `atomicAdd` 顺序非确定，浮点加法非结合律导致末位差异，不能用 `==` 比较
- `test_hash_table_overflow_retry`: 人为给定过小 capacity，验证重试逻辑

## 6. 非目标

- 多节点 shard_map 集成: 不在此 spec 中 (基础设施已预留)

## 7. 已知风险

| 风险 | 等级 | 缓解 |
|------|------|------|
| 哈希表溢出 (open addressing 无限 probe) | **高** | probe 上限 100 硬限；Python 层 overflow retry 未实现 |
| `compute_diagonal` 对角 term 仅占 1-10%，grid-stride loop 避免 10^12 线程启动 | **高** | 预过滤对角 term 子集 + grid-stride loop (~10^5 blocks)，不用 T×B 网格 |
| 结构化 config bytes 导致 MurmurHash3 聚集 | 中 | 所有哈希表统一使用 wyhash |
| L2 cache thrashing（configs 流驱逐 hash table 条目） | 中 | `__ldg()` 走 read-only cache |
| 重复 alloc/free 哈希表导致 HBM 碎片化 | 中 | `cudaMemPool` 预分配；apply_within 哈希表跨调用缓存（已预留，未启用） |
| 大量 terms 单次 kernel 内存压力 | 低 | 当前 terms 数 (<10^6) 单次 kernel 可行；未来可用 chunking |
| 纯 JAX fallback 与 CUDA 非 bit-exact (`atomicAdd` 顺序非确定) | 低 | 测试用 `allclose(rtol=1e-12)` |
| `exclude_configs` 线性扫描耗时 | 低 | 当前 exclude 集小，线性扫描可行；未来可用哈希表 |
