# Spec: Networks 子系统实现（MLP + Transformers 自回归 NQS）

**日期**: 2026-07-13
**状态**: approved

## 1. 目标

实现 `src/qmp/networks/` 子包，提供两种自回归神经量子态 (ANQS) ansatz：

- **MLP**: 每个 site 一个独立 MLP，逐 site 条件振幅 (arXiv 2109.12606)
- **Transformers**: 单一 causal decoder (dense FFN，无 MoE)

每种 backbone 提供三个粒子数对称性变体：

| 变体 | 守恒量 | site 单位 | 每 site 状态数 |
|------|--------|-----------|---------------|
| `WaveFunctionNormal` | 无 | 1 qudit | 任意 `physical_dim` |
| `WaveFunctionElectron` | 总电子数 | 1 qubit | 2 |
| `WaveFunctionElectronUpDown` | 自旋上/下电子数分别守恒 | 2 qubit (up,down) | 4 |

配套 JAX 版 `utility/bitspack.py`。

所有网络满足 `NetworkProto` 契约：`__call__(configs) -> psi`、`generate(batch_size, *, key)`、`generate_unique(batch_size, *, key)`。

## 2. 技术基调（已确认的设计决策）

1. **Flax nnx** API（非 linen）
2. Transformer decoder 使用 **dense FeedForward**，不含 DeepSeekMoE（无 shared/routed experts、centroid、负载均衡 bias）
3. `generate` 与 `generate_unique` **显式接收 PRNG key**（纯函数，无内部随机状态）
4. 顺带实现 **JAX 版 `utility/bitspack.py`**
5. 生成循环使用 **unrolled Python for**（sites 数在构造时静态已知，循环在 jit trace 期展开）
6. Normal 变体支持**任意 physical_dim**（qudit）
7. 测试为**内部自洽 unit test**，不与旧 PyTorch 实现数值对拍

## 3. 文件结构

```
src/qmp/utility/
├── __init__.py
└── bitspack.py                # JAX pack_int / unpack_int (size ∈ {1,2,4,8})

src/qmp/networks/
├── __init__.py                # re-export 六个 WaveFunction 类
├── _protocol.py               # NetworkProto Protocol
├── _autoregressive.py         # 共享逻辑：归一化、mask、gumbel-topk、采样
├── mlp.py                     # MLP backbone + 3 变体
└── transformers.py            # dense decoder backbone + 3 变体

tests/unit/
├── utility/
│   ├── __init__.py
│   └── test_bitspack.py
└── networks/
    ├── __init__.py
    ├── test_autoregressive.py
    ├── test_mlp.py
    └── test_transformers.py
```

## 4. 数值约定

- **float64**: log-amplitude / log-phase 累加与复振幅组装用 float64。JAX 默认 float32，需在 `src/qmp/__init__.py` 顶部启用 `jax.config.update("jax_enable_x64", True)`。这是包级数值策略（Hamiltonian 子系统同样使用 float64）。
- **config 编码**: bit-packed uint8 `[batch_size, n_qubytes]`，与 Hamiltonian 子系统输入格式一致。`n_qubytes = ceil(n_qubits * bits_per_site_element / 8)`。
- **psi 输出**: complex128，一维 `[batch_size]`。

## 5. bitspack 设计

`src/qmp/utility/bitspack.py`，纯 `jax.numpy` 实现，语义与旧 torch 版逐位对齐：

```python
def pack_int(array: Array, size: int) -> Array:
    """将 last_dim 上多个 size-bit 小整数打包进 uint8 字节 (LSB-first)。
    size ∈ {1,2,4,8}; elements_per_byte = 8 // size。
    尾部不足一字节的元素补零。输出 shape [..., ceil(last_dim / elements_per_byte)]。"""

def unpack_int(array: Array, size: int, last_dim: int) -> Array:
    """pack_int 的逆运算。输出 shape [..., last_dim]。"""
```

- shift = `arange(0, 8, size)`，`packed = sum(element << shift)`；`mask = (1 << size) - 1`。
- 输入输出均为 uint8。

## 6. `_autoregressive.py` — 共享逻辑

与具体 backbone 正交、三变体共用的纯函数：

### 6.1 条件振幅归一化

```python
def normalize_log_amplitude(log_amp: Array, axes: tuple[int, ...]) -> Array:
    """归一化局部条件 log-amplitude 使 Σ p = Σ exp(2·log_amp) = 1。
    param = 0.5 · log(Σ exp(2·log_amp))；返回 log_amp - param。"""
```

### 6.2 三种粒子数 mask

返回布尔张量，True 表示"可以在当前不完整构型后追加该状态"。违反约束的分支后续被赋 log-amplitude = -inf。

- `mask_normal`: 无约束（恒 True）。
- `mask_electron(partial_config, site_index, total_sites, electrons) -> [batch, 2]`:
  `add_hole = hole < total_sites - electrons`；`add_electron = electron < electrons`。
- `mask_electron_up_down(partial_config, site_index, total_sites, spin_up, spin_down) -> [batch, 2, 2]`:
  up/down 各自独立约束的逻辑与。索引 `[., up, down]`。

### 6.3 Gumbel top-K 束搜索单步

实现自回归 Gumbel top-K trick (arXiv 2408.07625 / Kool et al. 2020)：

```python
def gumbel_topk_step(
    parent_log_prob: Array,          # [beam] 累积无扰动 log p
    parent_perturbed: Array,         # [beam] 父节点条件扰动 log p (L̃)
    child_cond_log_prob: Array,      # [beam, n_states] 子节点条件 log p(x_i|x_<i)
    key: Array,
) -> tuple[Array, Array]:
    """返回 (child_log_prob [beam, n_states], child_perturbed [beam, n_states])。
    l = parent_log_prob[:,None] + child_cond_log_prob
    L = l + Gumbel(key)                       # G = -log(-log U)
    Z = max_over_states(L)                    # [beam, 1]
    L̃ = -log( exp(-parent_perturbed[:,None]) - exp(-Z) + exp(-L) )
    """
```

**数值护栏**: 无效分支（p=0，log p=-inf）与 padding 束槽用有限 sentinel `-1e30` 而非字面 `-inf`，配合 `jnp.where` 避免 `inf - inf = NaN`。束宽固定为 K，早期不足 K 用 sentinel 填充；只在全部 site 处理完后过滤一次有效条目。

### 6.4 multinomial 采样单步（generate 用）

```python
def sample_step(cond_log_prob: Array, key: Array) -> Array:
    """按 p = exp(2·cond_log_prob) 采样每个 batch 元素的下一状态。
    使用 jax.random.categorical（log 概率 = 2·cond_log_prob）。"""
```

## 7. 契约 `NetworkProto`

```python
class NetworkProto(Protocol):
    def __call__(self, configs: Array) -> Array:
        """configs: uint8 [batch, n_qubytes]。返回 psi: complex128 [batch]。"""

    def generate(self, batch_size: int, *, key: Array) -> tuple[Array, Array, Array]:
        """有放回采样。返回 (configs [n_unique, n_qubytes] uint8,
        psi [n_unique] complex128, counts [n_unique] int)。"""

    def generate_unique(self, batch_size: int, *, key: Array) -> tuple[Array, Array]:
        """无放回 (Gumbel top-K) 采样。返回 (configs [<=batch_size, n_qubytes] uint8,
        psi [<=batch_size] complex128)。"""
```

## 8. MLP 设计 (`mlp.py`)

- `MLP(nnx.Module)`: 多层 `nnx.Linear` + SiLU 激活。`dim_in == 0` 时退化为 bias-only 层（对应旧 `FakeLinear`，避免零输入维度）。
- 三变体各含：
  - `amplitude`: site 数个独立 MLP（第 i 个输入维度 = i × 每 site 元素数，输出 = 每 site 状态数）
  - `phase`: 单个 MLP（输入 = 全部 site，输出 = 1）
- `__call__`: unpack → 应用 `ordering_reversed` → 逐 site 并行算条件振幅 → 加 mask → 归一化 → gather 实际状态 → 累加 log-amplitude；phase 网络算总相位 → 组装 `exp(amp + i·phase)`。
- `generate` / `generate_unique`: unrolled Python for over sites，逐 site 扩展束/采样。

## 9. Transformers 设计 (`transformers.py`)

building blocks（均为 `nnx.Module`）：

- `Embedding`: 每个 position 一个 `[physical_dim, embedding_dim]` 查表；输入位置 `base` 偏移。
- `SelfAttention`: fused QKV Linear + causal mask + KV-cache（k/v 经 `jax.lax.stop_gradient` 截断历史梯度）。多头。
- `FeedForward`: `Linear(emb→hidden) → GELU → Linear(hidden→emb)`（dense，无 MoE）。
- `DecoderUnit`: pre-norm self-attention + 残差 + LayerNorm + dense FFN + 残差 + LayerNorm。
- `Transformers`: depth 层 `DecoderUnit` 堆叠。
- `Tail`: `Linear → GELU → Linear`，输出维度 = 2 × 每 site 状态数（amplitude + phase）。

三变体：
- config 移位（丢弃末 site，前置 BOS=0），使 position t 编码"前 t 个 site 的历史"。
- `__call__`: 并行处理全部 site（causal mask 保证无未来信息泄漏）。
- `generate*`: unrolled Python for，逐 token 处理，KV-cache 逐步 concat（每步形状为具体常量，jit 展开无碍）。

任意 physical_dim 在 Normal 变体：`Embedding` 状态数 = physical_dim，`Tail` 输出 = 2 × physical_dim。

## 10. ordering

`ordering` 参数：`+1`（正序）、`-1`（逆序）或自定义 `list[int]`。`__call__` 时用 `ordering_reversed` 将用户序还原为规范内部序；`generate*` 输出打包时用 `ordering` 还原为用户序。

## 11. 测试计划（内部自洽）

### 11.1 test_bitspack.py
- pack∘unpack 往返一致（各 size ∈ {1,2,4,8}）
- padding 边界（last_dim 非 elements_per_byte 整数倍）
- 已知小例手工验证位布局

### 11.2 test_autoregressive.py
- `normalize_log_amplitude` 后 Σ exp(2·x) = 1
- 三种 mask 粒子数约束正确（含手工小例）
- `gumbel_topk_step` 无 NaN（含无效分支 sentinel）
- gumbel-topk 束搜索唯一性 + K 上限

### 11.3 test_mlp.py / test_transformers.py（各三变体）
- `__call__` 输出 shape=[batch]、dtype=complex128
- **归一化自洽**: 小系统枚举全 Hilbert 空间，Σ|ψ|² = 1
- **粒子数守恒**: Electron/ElectronUpDown 变体对违反粒子数的 config 返回 ψ=0
- **forward/generate 自洽**: `generate_unique` 返回的 psi 与对同一 config 调 `__call__` 一致
- **generate_unique 唯一性**: 无重复 config，数量 ≤ batch_size
- **generate counts**: counts 之和 = batch_size；configs 唯一
- **PRNG 确定性**: 同 key 复现
- **ordering**: +1/-1/自定义 list 行为正确
- **任意 physical_dim**（Normal 变体）: physical_dim=3 等非 2 幂次可运行且归一化

## 12. 依赖

新增运行时依赖: `flax`（含 nnx）。

## 13. 非目标

- 多节点 shard_map 集成（接口保持纯函数 + 显式 key，为后续预留）
- 与旧 PyTorch 实现数值对拍
- MPS 网络（本 spec 不含）
- checkpoint 序列化（nnx 参数为标准 pytree，后续由 algorithms 层处理）

## 14. 已知风险

| 风险 | 等级 | 缓解 |
|------|------|------|
| Gumbel 条件截断 `inf - inf = NaN` | 高 | 有限 sentinel -1e30 + `jnp.where` 护栏 |
| unrolled 循环在大 sites 编译时间长 | 中 | 本 spec 目标 sites ≲ 100；后续可选 scan 优化 |
| x64 全局启用影响其他子系统 | 低 | 包级策略，Hamiltonian 亦用 float64；文档记录 |
| 任意 physical_dim 的 bitspack size 仅支持 {1,2,4,8} | 低 | physical_dim 状态编码 bit 数取 {1,2,4,8}；spec 限定 physical_dim ≤ 256 |
