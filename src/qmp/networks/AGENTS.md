# Networks 子系统设计

## 概述

Networks 子系统提供**变分波函数 ansatz**（神经量子态，NQS）。神经网络参数化量子态 |ψ⟩，供 VMC 优化。当前实现两种自回归骨架：

- **MLP** (`mlp.py`)：每个 site 一个独立 MLP，逐 site 预测条件振幅（arXiv:2109.12606）
- **Transformers** (`transformers.py`)：单一 causal decoder（dense FFN，无 MoE）

每种骨架有三个粒子数对称性变体：

| 变体 | 守恒量 | site 单位 | 每 site 状态数 |
|------|--------|-----------|---------------|
| `WaveFunctionNormal` | 无 | 1 qudit | 任意 `physical_dim` (≤256) |
| `WaveFunctionElectron` | 总电子数 | 1 qubit | 2 |
| `WaveFunctionElectronUpDown` | 自旋上/下电子数分别守恒 | 2 qubit (up,down) | 4 |

## 契约 (`_protocol.py`)

所有网络实现 `NetworkProto`：

- `__call__(configs) -> psi`：估值。configs 为 bit-packed uint8 `[batch, n_qubytes]`，返回 complex128 `[batch]`。
- `generate(batch_size, *, key) -> (configs, psi, counts)`：有放回采样，去重返回。`counts` 之和 = `batch_size`。
- `generate_unique(batch_size, *, key) -> (configs, psi)`：无放回采样（Gumbel top-K），返回 ≤ `batch_size` 个唯一构型。

**返回签名相对旧 PyTorch 版做了简化**：去掉了末尾的 `None` 占位（旧版返回 4 元组）。

## 设计原则

### 自回归分解与骨架无关

波函数分解为 ψ(x) = ∏ᵢ ψᵢ(xᵢ|x₍<ᵢ₎)。条件振幅的计算（骨架）与三件正交的事情解耦，全部集中在 `_autoregressive.py`：

1. **归一化** `normalize_log_amplitude`：局部条件振幅归一化使 Σexp(2·log_amp)=1
2. **粒子数 mask** `mask_electron` / `mask_electron_up_down`：把违反守恒的分支 log-amplitude 设 -inf
3. **采样** `sample_step`（multinomial）与 `gumbel_topk_step`（无放回束搜索）

新增骨架只需实现条件振幅计算 + 几个 variant hook（config 编解码、`_local_mask`），复用全部采样逻辑。

### Gumbel top-K 与骨架正交

无放回采样（arXiv:2408.07625, Kool et al. 2020）是纯粹的束搜索，独立于骨架：累积 log p → 加 Gumbel 噪声 → 条件截断 `L̃ = −log(exp(−L̃_parent) − exp(−Z) + exp(−L))` → 按 L̃ 取 top-K。

**NaN 护栏**：束宽固定为 K，早期不足或粒子数非法的槽用有限 sentinel `INVALID_LOG_PROB = -1e30`（而非字面 -inf），配合 `jnp.where` 避免条件截断中 `inf − inf = NaN`。只在全部 site 处理完后过滤一次有效槽。

### 生成循环用 unrolled Python for

sites 数在构造时静态已知，生成循环逐 site 展开。各步形状虽不同但都是具体常量。这也是 MLP 变体唯一可行方式（每 site 是独立模块，输入维度随 i 增长，无法 scan）。

注意可 jit 性：`__call__`（估值）可以 jit。但 `generate` / `generate_unique` 末尾用 `jnp.unique` / 布尔过滤产生**动态形状**，因此**整体不可 jit**，按 eager 执行——unrolled 循环只是让每步以具体形状运行，并非把整个函数塞进一次 jit。

### Transformer 的 KV-cache 增量解码

- `__call__`（估值）：一次并行前向 + causal mask。
- `generate*`（采样）：增量 KV-cache 解码。每步只喂新 token，attention 复用缓存的 key/value（`nnx.MultiHeadAttention` 的 `decode=True` + `init_cache`）。
- **束搜索中的 cache 重排**：`generate_unique` 每步剪枝后，按父束索引 `parents = selected // states` 沿 batch 轴重排每层 attention 的 `cached_key`/`cached_value`。这保证缓存跟随束的祖先路径。
- 因为生成是纯前向（无 autograd 磁带），旧代码的 `detach` 不需要。
- **副作用提示**：`generate*` 会在模块上通过 `init_cache` 建立 `nnx.Cache` 变量并原地更新（KV-cache 本质有状态）。这些 Cache 变量在调用后仍留存于模块（每次调用会重新分配、正确按当前 batch_size 重建）。它们属于 `nnx.Cache` 类别，**不属于 `nnx.Param`**，故按 `nnx.Param` 过滤的优化器/梯度不受影响；但 `nnx.state(net)`（全量）会包含它们。这是唯一偏离"纯函数优于副作用"原则的地方，源于 nnx KV-cache 的有状态设计。
- 正确性验证：`generate_unique` 返回的 psi 与对同一 config 调 `__call__` 数值一致（见 `test_generate_unique_matches_parallel_amplitude`）。

### 初始化：输出层零初始化

**输出头（MLP 的 amplitude/phase 末层、Transformer 的 Tail 末层）零初始化**，使初始条件分布近似均匀（最大熵）。这是 VMC 的中性起点——避免网络初始就偏向某个随机构型。

不要用 `normal(stddev=1.0)` 初始化输出层：实测会产生强尖峰的随机初态（maxprob≈0.5，远离均匀的 0.17）。位置嵌入用小 stddev（0.02），隐藏层用 flax 默认（lecun/glorot）。

### float64

log-amplitude、phase 全程 float64 累加（根 AGENTS.md 的全局 x64 策略）。网络参数用 `param_dtype=jnp.float64` 显式声明。

## nnx 使用注意（flax 0.12+）

- 模块列表用 `nnx.List(...)`，不能直接用 Python `list`（会报 static/data 错误）。
- `nnx.Linear(0, ...)` 会除零崩溃；零输入维度的层需自建 bias-only 实现（见 `mlp._Linear`）。
- KV-cache 状态是 `nnx.Cache` 变量；重排用直接属性赋值 `attention.cached_key = nnx.Cache(...)`。
- `set_attributes(decode=...)` 递归切换所有子模块的 decode 标志。

## ordering

`ordering` 参数：`+1`（正序）、`-1`（逆序）或自定义 `list[int]` 排列。`__call__` 用 `ordering_reversed` 将用户序还原为规范内部序；`generate*` 输出打包时用 `ordering` 还原为用户序。

## 未包含（后续工作）

- MPS 网络
- 多节点 shard_map 集成（接口已保持显式 key + 纯 pytree 参数）
- checkpoint 序列化（nnx 参数为标准 pytree，由 algorithms 层处理）
- 生成时的显存分块（`block_num`）—— 暂时移除

## 测试

- `test_autoregressive.py`：归一化（含逐行独立、常数平移）、mask、Gumbel 步（无 NaN、上界性质、无效分支排序下沉）、采样（禁止态、确定性、经验 Born）
- `test_mlp.py` / `test_transformers.py`：各三变体的
  - 契约（shape/dtype）、全空间归一化、粒子数守恒
  - 采样正确性：generate/generate_unique 自洽、唯一性、穷尽性、经验频率≈Born、determinism
  - 数学性质：batch 不变性、jit==eager、config 往返、初始态实数、梯度流、相位分离（MLP）、因果性（Transformer）
  - KV-cache 隔离（Transformer：不污染 Param、重复/变 batch 一致）
  - 边界与架构变体：electrons=0/sites、ordering、任意 physical_dim、单 site、非对称自旋、多/空隐层、depth=1/heads=1

测试原则：只做内部自洽的单元测试（不与旧 PyTorch 对拍）。避免昂贵的统计性网络级测试——采样分布正确性在 `_autoregressive` 层廉价验证；VMC 收敛、多卡、真实 Hamiltonian 联算属集成测试范畴，不放在此处。
