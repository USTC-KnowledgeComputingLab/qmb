# Spec: Hamiltonian 子系统实现

**日期**: 2026-06-17
**状态**: approved

## 1. 目标

实现 `src/qmp/hamiltonian/` 子包，包含：

- **位掩码预处理** (`prepare`): 纯 Python 将费米子哈密顿量字典转为 (create_mask, annihilate_mask, flip_mask, parity_mask, parity_const, coef) 表示
- **四个 CUDA kernel**: `compute_diagonal_within_subspace`, `apply_within_subspace`, `find_all_relative_configs`, `find_topk_relative_configs` — 通过 JAX FFI 接入
- **纯 JAX fallback**: CPU/CI 环境下无 GPU 可用时，用 `jax.jit` 实现相同功能
- **JIT 编译与缓存**: CUDA kernel 按需编译 (.cu → .so)，缓存于 `~/.cache/qmp/`
- **pytest 测试**: prepare 正确性 + fallback 端到端 + CUDA 回归

## 2. 文件结构

```
src/qmp/hamiltonian/
├── AGENTS.md                    # 子系统设计文档 (已存在, 待更新)
├── __init__.py                  # re-export Hamiltonian
├── _hamiltonian.py              # Hamiltonian 类 + FFI 注册 + CUDA/fallback 路由
├── _hamiltonian_prepare.py      # 纯 Python 位掩码预处理 → JAX arrays
├── _hamiltonian_jax.py          # 纯 JAX fallback 实现 (四个操作, @jax.jit)
├── _hamiltonian_cuda.cu         # CUDA kernel (四个操作, 模板化 n_qubytes/particle_cut/max_op_number)
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

| 操作 | Python 方法 | XLA FFI target | C++ handler | 语义 |
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

**算法**:

对每个 term，将产生/湮灭算符分类后按正规序排列作用序列:
1. 逐算符计算对初始构型的约束条件 (flip_bit ⊕ (0 if creation else 1))
2. 累加 JW 奇偶性常数 (const ⊕= popcount(flip_before & low_mask(idx)))
3. 更新翻转掩码 (flip ^= 1 << idx)
4. 检查约束一致性，不一致则跳过 (项恒为零)

Python int 天然支持无限精度, n_qubits > 64 也无问题。`popcount` 用 `int.bit_count()`。

### 3.2 `_hamiltonian_cuda.cu` — CUDA kernel

**模板参数**:
- `N_QUBYTES` (编译期宏): ceil(n_qubits/8)
- `PARTICLE_CUT` (编译期宏): 1 (fermi) 或 2 (bose2)
- `MAX_OP_NUMBER` (编译期宏): 每 term 最大算符数 (二体相互作用 = 4)

**四个 handler** (`XLA_FFI_DEFINE_HANDLER_SYMBOL`):
- `ComputeDiagonalWithinSubspace` — 对角元 (完整实现)
- `ApplyWithinSubspace` — 稀疏 H·psi 投影 (forward + backward 方向)
- `FindAllRelativeConfigs` — cuCollections static_map 哈希表去重
- `FindTopKRelativeConfigs` — 哈希表 + 阈值加速 + 周期性 CUB compact

**工具函数** (device):
- `is_applicable(config, create_mask, annihilate_mask, n_qubytes)` — 位 AND/OR+NOR check
- `apply_flip(config, flip_mask, n_qubytes)` — XOR
- `jw_parity(config, parity_mask, parity_const, n_qubytes)` — popcount + XOR

**编译期分支**: 对 n_qubits <= 64 可使用 64-bit 寄存器优化；n_qubits > 64 逐字节。

### 3.3 `_hamiltonian_cuda_loader.py` — JIT 编译与缓存

```python
def load_cuda_module(n_qubytes: int, particle_cut: int, max_op_number: int) -> ctypes.CDLL:
    key = f"qmp_hamiltonian_{n_qubytes}_{particle_cut}_{max_op_number}"
    cache_dir = platformdirs.user_cache_path("qmp", "kclab") / key
    so_path = cache_dir / "lib.so"

    if not so_path.exists():
        # 用 nvcc 编译 _hamiltonian_cuda.cu
        subprocess.run([
            "nvcc", "-shared", "-Xcompiler", "-fPIC",
            f"-I{jaxlib.get_include_dir()}",
            f"-DN_QUBYTES={n_qubytes}",
            f"-DPARTICLE_CUT={particle_cut}",
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

四个 `@jax.jit` 函数，语义与 CUDA kernel 完全一致:

```python
@jax.jit
def compute_diagonal_within_subspace(configs, create_mask, annihilate_mask, flip_mask, parity_mask, parity_const, coef):
    # 对每个 (term, config): 可作用性检查 + 翻转 = 0? + JW parity + 累加 coef
    ...

@jax.jit
def apply_within_subspace(configs_i, psi_i, configs_j, create_mask, annihilate_mask,
                          flip_mask, parity_mask, parity_const, coef, direction=0):
    ...

@jax.jit
def find_all_relative_configs(configs_i, psi_i, configs_exclude, create_mask, annihilate_mask,
                              flip_mask, parity_mask, parity_const, coef, hash_capacity):
    ...

@jax.jit
def find_topk_relative_configs(configs_i, psi_i, count_selected, configs_exclude, create_mask,
                                annihilate_mask, flip_mask, parity_mask, parity_const, coef):
    ...
```

### 3.5 `_hamiltonian.py` — Hamiltonian 类 + FFI 路由

```python
class Hamiltonian:
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
- 与 test_fallback 相同输入，assert CUDA 输出 == JAX fallback 输出 (tol=1e-10)

## 6. 非目标

- SpinSeparatedHamiltonian: 不实现
- 多节点 shard_map 集成: 不在此 spec 中 (基础设施已预留)
- n_qubits > 64 的逐字节 CUDA kernel: n_qubits <= 64 的 64-bit 寄存器版本优先。多字节扩展后续迭代
