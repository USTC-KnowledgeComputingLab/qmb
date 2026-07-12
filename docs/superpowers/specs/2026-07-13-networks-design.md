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
3. `generate` 与 `generate_unique` **显式接收 PRNG key**（随机性无内部状态；注意 Transformer 生成会在模块上留下 KV-cache 变量，见 §9）
4. 顺带实现 **JAX 版 `utility/bitspack.py`**
5. 生成循环使用 **unrolled Python for**（sites 数在构造时静态已知，逐 site 展开）。`__call__`（估值）用 `@nnx.jit` 修饰；每步前向（MLP `_conditional_log_amplitude` / Transformer `_decode_step`）用 `nnx.jit` + 静态 `site_index` 修饰并跨步缓存。`generate` / `generate_unique` 因末尾用 `jnp.unique` / 布尔过滤产生动态形状，**整体不可 jit**，保持 eager 并调用上述被 jit 的静态方法
6. Normal 变体支持**任意 physical_dim**（qudit）
7. 测试为**内部自洽 unit test**，不与旧 PyTorch 实现数值对拍

## 3. 文件结构

```
src/qmp/utility/
├── __init__.py
└── bitspack.py                # JAX pack_int / unpack_int (size ∈ {1,2,4,8})

src/qmp/networks/
├── __init__.py                # re-export NetworkProto + mlp / transformers 模块
├── AGENTS.md                  # 子系统设计与局部原则
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

- **float64**: log-amplitude / log-phase 累加与复振幅组装用 float64。JAX 默认 float32，需在 `src/qmp/__init__.py` 中（导入后、任何数值运算前）调用 `jax.config.update("jax_enable_x64", True)`。这是包级数值策略（Hamiltonian 子系统同样使用 float64）。
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
def normalize_log_amplitude(log_amplitude: Array, axis: int = -1) -> Array:
    """归一化局部条件 log-amplitude 使 Σ p = Σ exp(2·log_amp) = 1。
    log_partition = 0.5 · logsumexp(2·log_amplitude)；返回 log_amplitude - log_partition。"""
```

`apply_mask(log_amplitude, mask)`: 把 `mask` 为 False 的态设为 `-inf`（零概率）。

### 6.2 两种粒子数 mask

返回布尔张量，True 表示"可以在当前不完整构型后追加该状态"。违反约束的分支经 `apply_mask` 赋 log-amplitude = -inf。Normal 变体无约束，直接内联 `jnp.ones(...)`（不设独立函数）。

- `mask_electron(electron_count, sites_filled, total_sites, electrons) -> [..., 2]`:
  `can_add_hole = hole_count < total_sites - electrons`；`can_add_particle = electron_count < electrons`。
- `mask_electron_up_down(up_count, down_count, sites_filled, total_sites, spin_up, spin_down) -> [..., 2, 2]`:
  up/down 各自独立约束的逻辑与。索引 `[., up, down]`。

### 6.3 Gumbel top-K 束搜索单步

实现自回归 Gumbel top-K trick (arXiv 2408.07625 / Kool et al. 2020)：

```python
def gumbel_topk_step(
    parent_log_prob: Array,          # [beam] 累积无扰动 log p
    parent_perturbed: Array,         # [beam] 父节点条件扰动 log p (L̃)
    parent_valid: Array,             # [beam] 是否为有效（非 padding）束槽
    conditional_log_prob: Array,     # [beam, n_states] 子节点条件 log p(x_i|x_<i)，禁止态为 -inf
    key: Array,
) -> tuple[Array, Array, Array]:
    """返回 (child_log_prob, child_perturbed, child_valid)，均为 [beam, n_states]。
    l = parent_log_prob[:,None] + conditional_log_prob
    L = l + Gumbel(key)                       # G = -log(-log U)
    Z = max_over_states(L)                    # [beam, 1]
    L̃ = -log( exp(-parent_perturbed[:,None]) - exp(-Z) + exp(-L) )
    无效子节点（parent 为 padding 或 conditional 为 -inf）标记 child_valid=False 并置 sentinel。
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

- `_MLP(nnx.Module)`: 多层 `_Linear` + SiLU 激活；`zero_output=True` 时末层零初始化。
- `_Linear`: 包装 `nnx.Linear`；`in_features == 0` 时退化为 bias-only 层（对应旧 `FakeLinear`，因 `nnx.Linear(0, …)` 会除零崩溃）。
- 三变体各含：
  - `amplitude`: site 数个独立 `_MLP`（第 i 个输入维度 = i × 每 site 元素数，输出 = 每 site 状态数），末层零初始化
  - `phase`: 单个 `_MLP`（输入 = 全部 site，输出 = 1），末层零初始化
- `__call__`: unpack → 应用 `ordering_reversed` → 逐 site 算条件振幅 → 加 mask → 归一化 → gather 实际状态 → 累加 log-amplitude；phase 网络算总相位 → 组装 `exp(amp + i·phase)`。
- `generate` / `generate_unique`: unrolled Python for over sites，逐 site 扩展束/采样。

## 9. Transformers 设计 (`transformers.py`)

building blocks（均为 `nnx.Module`）：

- `_PositionalEmbedding`: 每个 position 一个 `[physical_dim, embedding_dim]` 查表；`position` 参数选择起始行，使单个增量 token 能在其真实位置被嵌入。
- 注意力: 直接用 `nnx.MultiHeadAttention`（fused QKV，`decode=False` 并行 + causal mask；`decode=True` 增量 + KV-cache）。生成为纯前向，无梯度磁带，因此无需截断历史梯度（无 `stop_gradient`）。
- `_FeedForward`: `Linear(emb→hidden) → GELU → Linear(hidden→emb)`（dense，无 MoE）。
- `_DecoderUnit`: pre-norm self-attention + 残差 + LayerNorm + dense FFN + 残差。
- `_Transformers`: depth 层 `_DecoderUnit` 堆叠；并行模式建 causal mask，decode 模式靠 KV-cache 无需 mask。
- `_Tail`: `Linear → GELU → Linear`，输出维度 = 2 × 每 site 状态数（amplitude + phase）；末层零初始化。

三变体：
- config 移位（丢弃末 site，前置 BOS=0），使 position t 编码"前 t 个 site 的历史"。
- `__call__`: 并行处理全部 site（causal mask 保证无未来信息泄漏）。
- `generate*`: unrolled Python for，**增量 KV-cache 解码**。每步只喂新 token，attention 复用缓存的 key/value（`nnx.MultiHeadAttention` 的 `decode=True` + `init_cache`）。`generate_unique` 每步剪枝后按父束索引 `parents = selected // states` 沿 batch 轴重排每层 attention 的 cache，使缓存跟随束的祖先路径。生成为纯前向，无需 `detach`。

任意 physical_dim 在 Normal 变体：`Embedding` 状态数 = physical_dim，`Tail` 输出 = 2 × physical_dim。

### 初始化

输出头（MLP amplitude/phase 末层、Transformer Tail 末层）**零初始化**，使初始条件分布近似均匀（最大熵），为 VMC 提供中性起点。位置嵌入用小 stddev（0.02），隐藏层用 flax 默认。实测 `normal(stddev=1.0)` 输出层会产生强尖峰随机初态，不合适。

## 10. ordering

`ordering` 参数：`+1`（正序）、`-1`（逆序）或自定义 `list[int]`。`__call__` 时用 `ordering_reversed` 将用户序还原为规范内部序；`generate*` 输出打包时用 `ordering` 还原为用户序。

## 11. 测试计划（内部自洽）

### 11.1 test_bitspack.py
- pack∘unpack 往返一致（各 size ∈ {1,2,4,8}，对齐与非对齐）
- padding 边界（last_dim 非 elements_per_byte 整数倍）+ 截断丢弃 padding
- 已知小例手工验证位布局（LSB-first）、多维输入、dtype 保持、各 size 最大值往返
- 错误路径：非 uint8 输入、非法 size ∈ {0,3,5,7,16} 均报错

### 11.2 test_autoregressive.py
- `normalize_log_amplitude`：Σ exp(2·x) = 1（含 mask 情形）、逐行独立归一化、= 逐行减常数
- 两种 mask 粒子数约束正确（含手工小例；Normal 无约束不单独测）
- `gumbel_topk_step`：无 NaN（含无效分支 sentinel）、扰动值不超过父上界、累积 log-prob 正确、padding 父节点全无效、无效子节点排序下沉
- `sample_step`：尊重禁止态、同 key 确定性、经验频率 ≈ Born 分布

（gumbel-topk 束搜索的唯一性 + K 上限 + 穷尽性在网络级 `test_mlp.py` / `test_transformers.py` 中验证。）

### 11.3 test_mlp.py / test_transformers.py（各三变体）

**契约与归一化**
- `__call__` 输出 shape=[batch]、dtype=complex128
- 归一化自洽：小系统枚举全 Hilbert 空间，Σ|ψ|² = 1
- 粒子数守恒：Electron/ElectronUpDown 对违反粒子数的 config 返回 ψ=0

**采样正确性**
- forward/generate 自洽：`generate_unique` / `generate` 返回的 psi 与 `__call__` 一致
- generate_unique：唯一性、数量 ≤ batch_size、穷尽性（束宽 ≥ 支持集时返回精确支持集）、超 K 时封顶到支持集大小、strict 子集概率 < 1
- generate：counts 之和 = batch_size、configs 唯一、经验频率 ≈ Born 分布
- PRNG 确定性（generate 与 generate_unique）、不同 key 给出不同样本

**数学性质**
- batch 不变性（逐个 vs 批量估值一致）、jit == eager
- config 编解码往返恒等
- 初始态为实数（输出层零初始化）、梯度流经 `__call__` 有限且非零
- Born 分布不受 phase 网络扰动影响（MLP 专属，相位/振幅分离）
- 因果性：早期 site 条件概率不依赖后续 site（Transformer）

**KV-cache 隔离（Transformer 专属）**
- `generate*` 不污染 `nnx.Param`、重复生成一致、变化 batch size 正确重建 cache

**边界与架构变体**
- 守恒边界：electrons=0（真空）、electrons=sites（全占据）
- ordering：+1/-1/自定义 list 行为正确 + 输出往返
- 任意 physical_dim（Normal）：physical_dim=3、4 归一化
- 单 site、非对称 spin_up≠spin_down、多隐层 / 空隐层（MLP）、depth=1/heads=1（Transformer）

## 12. 依赖

新增运行时依赖: `flax`（含 nnx）。

## 13. 非目标

- 多节点 shard_map 集成（接口保持显式 key 与纯 pytree 参数，为后续预留）
- 与旧 PyTorch 实现数值对拍
- MPS 网络（本 spec 不含）
- checkpoint 序列化（nnx 参数为标准 pytree，后续由 algorithms 层处理）
- 生成时的显存分块（`block_num`）

## 14. 已知风险

| 风险 | 等级 | 缓解 |
|------|------|------|
| Gumbel 条件截断 `inf - inf = NaN` | 高 | 有限 sentinel -1e30 + `jnp.where` 护栏 |
| unrolled 循环在大 sites 编译时间长 | 中 | 每步前向按 site 各编译一次并缓存；`generate*` 不整体 jit，动态形状部分保持 eager |
| x64 全局启用影响其他子系统 | 低 | 包级策略，Hamiltonian 亦用 float64；文档记录 |
| 任意 physical_dim 的 bitspack size 仅支持 {1,2,4,8} | 低 | physical_dim 状态编码 bit 数取 {1,2,4,8}；spec 限定 physical_dim ≤ 256 |
| Transformer `generate*` 在模块留下 KV-cache 变量（side effect） | 低 | Cache 属 `nnx.Cache` 非 `nnx.Param`，优化器/梯度不受影响；每次调用重建 |
