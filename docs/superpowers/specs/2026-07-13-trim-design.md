# Spec: TrimCI × HAAR 混合算法 (trim.py)

**日期**: 2026-07-13
**状态**: in-progress

## 1. 目标

新增一个 action `"trim"`，把 TrimCI（*From Random Determinants to the Ground State*, arXiv:2511.14734）的
**expansion + 两级 trim** 选态引擎与 HAAR 的**神经量子态 (NQS) 拟合回环**结合起来。

一个 cycle 的流程：

```
网络采样 → 多跳全局扩展(top-K) → 局部随机分块对角化(local trim)
→ 合并存活构型 → 全局对角化(global trim) → 构造 |ψ|² 目标
→ 梯度拟合回网络 → 更新 pool → checkpoint
```

## 2. 与 HAAR 的关系

- `haar.py` **保持不动**，用于对照实验。
- `trim.py` 从 `haar.py` **import 复用** 以下 helper（不复制）：
  `_apply_hamiltonian`、`_DynamicLanczos`、`_local_optimize`、`_sample_from_pool`、
  `_merge_pools`、`_save_checkpoint`、`_load_checkpoint`、`_AVAILABLE_LOSSES`、`KrylovBasisStrategy`。
- 与 HAAR 的核心区别：HAAR 用单个 Lanczos（可自适应扩基）直接得到目标；trim 用
  **显式 expansion 阶段 + 两级分块/全局对角化 trim** 得到目标。两者的第 5/6/7 步（目标构造、
  拟合回网络、pool/checkpoint）完全一致。

## 3. TrimCI 算法映射（论文 → 本实现）

| TrimCI 论文 | 本实现 | 说明 |
|---|---|---|
| Expansion 多跳 `\|H_ij c_j\|>θ`, `max_rounds` | `find_topk_relative_configs` 迭代 `max_rounds` 次 | 全局 top-K 幅度筛选代替动态阈值 θ |
| 跳间系数 `c_j` | `apply_within_subspace` 得到的 `H·ψ` 一阶振幅 | 真正多跳扩散（比论文单边 `\|H_ij c_j\|` 更精确） |
| Local trim: 随机分块 + 分块对角化 + 各留 top-kₐ | `jax.random.permutation` 随机分组 + 每组小 Lanczos + 每组 top-`local_keep_count` | 廉价、无偏、保多样性 |
| Global trim: 全局对角化 + top-k_b | 全局 Lanczos + 按 `\|psi\|` 取 top-`core_keep_count` | 变分上界 |
| 首轮少留 (`first_cycle_keep_size`) | 首轮 core 取 top-`first_cycle_keep_size` | 随机初始化时先滤 |

**对角化器**：论文用 Davidson；本实现复用 HAAR 的 `_DynamicLanczos`（Lanczos 也能求最低本征对）。
local trim 与 global trim **都用 Lanczos**（用户决策）。

## 4. 配置 dataclass

```python
@dataclass
class TrimConfig:
    model: SubConfigRef | None = None
    network: SubConfigRef | None = None

    # sampling（与 haar 一致）
    sampling_count_from_network: int = 1024
    sampling_count_from_pool: int = 1024

    # expansion（多跳全局 top-K）
    max_rounds: int = 4            # 多跳次数（论文 max_rounds）
    pool_core_ratio: int = 10      # 每跳新增量 = ratio × 当前 pool 大小

    # local trim（随机分块）
    num_groups: int = 10           # 随机分组数（论文 num_groups）
    local_keep_count: int = 64     # 每组保留的绝对 top-kₐ
    local_lanczos_steps: int = 16  # 每块 Lanczos 步数

    # global trim
    global_lanczos_steps: int = 32
    krylov_stop_norm: float = 1e-8
    krylov_random_period: int = 31
    krylov_state_count: int = 1
    core_keep_count: int = 128     # global 后取 top-k_b 作为下轮 core
    first_cycle_keep_size: int = 10  # 首轮少留

    # local optimization（与 haar 一致）
    loss_name: str = "sum_filtered_angle_scaled_log"
    local_max_steps: int = 10000
    local_stop_loss: float = 1e-8

    # checkpoint / cycle control
    checkpoint_path: str | None = None
    checkpoint_interval: int = 1
    max_cycles: int = -1           # -1 = 无限循环, >0 = 跑 N 轮后返回
```

## 5. 状态 dict

镜像 HAAR，顶层 key 改为 `"trim"`：

```python
state = {
    "trim": {
        "global": 0,          # cycle 计数
        "local": 0,           # 最近一轮 local optimize 的 step 数
        "pool": (configs, psi, counts),   # 或 None（首轮）
        "excited": [(E, configs, psi), ...],
    },
}
```

- `pool`: `(configs: [M,Q] uint8, psi: [M] complex128, counts: [M] float64)`
- `_sample_from_pool` 只用 `configs, psi`；`counts` 为兼容占位（`jnp.ones_like(psi.real)`）。
- `run()` 把最终 state 存到 `self._state`，便于测试在不写盘的情况下直接检查（与 haar 一致）。

## 6. 核心函数

### 6.0 `_unique_configs` — 构型去重

```python
def _unique_configs(configs):
    """位打包构型去重，保留首次出现顺序。

    镜像 haar._merge_pools 的 4 字节对齐技巧，把 uint8 行 view 成 uint32 再 jnp.unique。
    """
    n_bytes = configs.shape[1]
    padded = n_bytes if n_bytes % 4 == 0 else ((n_bytes // 4) + 1) * 4
    work = configs if n_bytes == padded else jnp.pad(configs, ((0, 0), (0, padded - n_bytes)))
    flat = work.reshape(work.shape[0], -1).view(jnp.uint32)
    _, idx = jnp.unique(flat, axis=0, return_index=True)
    return configs[jnp.sort(idx)]
```

### 6.1 `_expand_pool` — 多跳全局扩展

```python
def _expand_pool(model, core_configs, core_psi, max_rounds, pool_core_ratio):
    """从 core 出发做 max_rounds 次全局 top-K 扩展，跳间用 H·ψ 传播权重。

    返回 (pool_configs, pool_psi)：pool_psi 是最后一跳后 H·ψ 的一阶振幅。
    """
    pool_configs, pool_psi = core_configs, core_psi
    for _ in range(max_rounds):
        psi_real = jnp.stack([pool_psi.real, pool_psi.imag], axis=1)   # [n, 2]
        count = pool_core_ratio * pool_configs.shape[0]
        new_c = model.find_topk_relative_configs(pool_configs, psi_real, count, pool_configs)
        if new_c.shape[0] == 0:
            break
        # 去重：top-K 在小系统会返回零填充行，须去重以免 pool 混入重复/零构型
        new_pool = _unique_configs(jnp.concatenate([pool_configs, new_c], axis=0))
        hpsi = model.apply_within_subspace(pool_configs, psi_real, new_pool)  # [|new_pool|, 2]
        pool_configs = new_pool
        pool_psi = hpsi[:, 0] + 1j * hpsi[:, 1]
    return pool_configs, pool_psi
```

**关键**：
- psi 传 `[batch, 2]` real/imag（与 kernel 契约一致；规避 haar `_extend` 的复数格式隐患）。
- `find_topk_relative_configs` 的 `configs_exclude` 传 `pool_configs`，只返回新增构型。
- 每跳用 `_unique_configs`（镜像 `haar._merge_pools` 的 4 字节对齐 + `jnp.unique` 技巧）
  去重：`find_topk_relative_configs` 在候选不足时会返回零填充行。
- 跳间 `apply_within_subspace(configs_i=pool_configs, psi_i=psi_real, configs_j=new_pool)`
  把振幅投影到扩大后的 pool，作为下一跳权重（真正多跳扩散）。

### 6.2 `_local_trim` — 随机分块 Lanczos

```python
def _local_trim(model, pool_configs, pool_psi, num_groups, keep_count,
                lanczos_steps, stop_norm, random_period, key):
    """随机分组 → 每组小 Lanczos 取最低 Ritz → 每组按 |c| 留 top-keep_count。

    返回合并去重后的存活 (configs, psi)。
    """
    n = pool_configs.shape[0]
    perm = jax.random.permutation(key, n)
    groups = jnp.array_split(perm, num_groups)

    survived = None
    for group_idx in groups:
        if group_idx.shape[0] == 0:
            continue
        block_c = pool_configs[group_idx]
        block_p = pool_psi[group_idx]
        # 步数上界为 block 维度：迭代超过子空间维度会耗尽 Krylov 空间，
        # 使 w 归零、随机注入护栏除以零范数（NaN）。小 trim block 极易触发。
        block_steps = min(lanczos_steps, max(block_c.shape[0] - 1, 1))
        lanczos = _DynamicLanczos(
            model=model, configs=block_c, psi=block_p,
            max_steps=block_steps, stop_norm=stop_norm,
            random_period=random_period, extend_count=0,
            strategy=KrylovBasisStrategy.FIXED, state_count=1,
        )
        results = list(lanczos.run())          # 取最后一步（最收敛）
        _e, cfg, ritz = results[-1][0]
        k = min(keep_count, cfg.shape[0])
        top = jnp.argsort((ritz.conj() * ritz).real)[::-1][:k]
        kept_c, kept_p = cfg[top], ritz[top]
        survived = (kept_c, kept_p) if survived is None else _merge_pools(survived[0], survived[1], kept_c, kept_p)
    assert survived is not None
    return survived
```

**注意**：
- `jnp.array_split` 用 Python 循环遍历静态 `num_groups` 个块（组数在 config 已知）。
- 每块的初始向量用 pool 的对应振幅子段（有物理信息、收敛快）。
- 每块 `max_steps` 上界取 `block_dim - 1`，防止小 block 上 Lanczos 迭代超出子空间维度
  触发零范数除法（NaN）。

### 6.3 `_global_trim` — 全局 Lanczos

复用 `_DynamicLanczos`（`FIXED`, `extend_count=0`, `state_count=krylov_state_count`），
遍历 `run()` 取最后一轮 `results`。下轮 core = 按 `|psi|` 取 top-`core_keep_count`
（首轮用 `first_cycle_keep_size`）。返回 `results` 供第 5 步目标构造。

### 6.4 目标构造 / 拟合 / pool 更新

与 haar.py:351-380 逐行一致，只改 state key 为 `"trim"`：

- 目标：多态 `|ψ|²` 累加 → 开方 → 按 `argmax(|target|)` 归一化 → `target_psi, max_idx`。
- 拟合：`_local_optimize(network, configs, target_psi, max_idx, loss_fn, local_max_steps, local_stop_loss)`。
- pool：`state["trim"]["pool"] = (configs, psi, jnp.ones_like(psi.real))`。

## 7. 主循环伪代码

```python
def run(self) -> None:
    if model is None or network is None: return
    loss_fn = _AVAILABLE_LOSSES[config.loss_name]
    state = load_checkpoint() or _init_state()
    cycle = state["trim"]["global"]
    start = cycle
    while max_cycles < 0 or cycle < start + max_cycles:
        key = jax.random.key(cycle * sampling_count_from_network)

        # 1. 采样
        c_net, p_net = network.generate_unique(sampling_count_from_network, key=key)
        c_pool, p_pool = _sample_from_pool(state["trim"]["pool"], sampling_count_from_pool, fold_in(key,1))
        core_c, core_p = _merge_pools(c_net, p_net, c_pool, p_pool)

        # 2. Expansion
        pool_c, pool_p = _expand_pool(model, core_c, core_p, max_rounds, pool_core_ratio)

        # 3. Local trim
        surv_c, surv_p = _local_trim(model, pool_c, pool_p, num_groups, local_keep_count,
                                     local_lanczos_steps, krylov_stop_norm, krylov_random_period, fold_in(key,2))

        # 4. Global trim
        lanczos = _DynamicLanczos(model, surv_c, surv_p, global_lanczos_steps, krylov_stop_norm,
                                  krylov_random_period, extend_count=0, strategy=FIXED, state_count=krylov_state_count)
        results = last of lanczos.run()
        _e0, configs, psi = results[0]
        keep = first_cycle_keep_size if cycle == 0 else core_keep_count
        # core 选择：按 |psi| top-keep 用于下轮 pool 记忆
        state["trim"]["excited"] = results

        # 5. Target
        target_psi, max_idx = build_target(results)

        # 6. Local optimize
        _p, _o, step = _local_optimize(network, configs, target_psi, max_idx, loss_fn, local_max_steps, local_stop_loss)
        state["trim"]["local"] = step

        # 7. pool + checkpoint
        core_sel = top-keep by |psi| over (configs, psi)
        state["trim"]["pool"] = (core_sel_c, core_sel_p, ones)
        state["trim"]["global"] = cycle + 1
        cycle += 1
        if cycle % checkpoint_interval == 0: save_checkpoint(...)
```

## 8. 注册

```python
# trim.py 底部
action_config_dict["trim"] = TrimConfig
action_class_dict["trim"] = Trim
```

同时在 `src/qmp/algorithms/__init__.py` 触发导入（与 haar/demo 一致）。

## 9. 依赖

- `jax`, `jax.numpy`, `flax.nnx`, `optax`（间接，通过复用的 `_local_optimize`）
- 从 `qmp.algorithms.haar` import 复用 helper
- 无新增第三方依赖

## 10. 非目标

- COO 轨道优化（论文二）：正交的一层，不在此 scope。
- PT2 微扰修正与外推：后续工作。
- 多次随机初始化并行 run 取最优（论文 ensemble 策略）：后续工作。
- Davidson 对角化器：复用 Lanczos。
- Multi-GPU shard_map：单卡优先。

## 11. 验证

1. `uv run ruff format . && uv run ruff check . && uv run ty check src tests`（零抑制）。
2. 单元测试 `tests/unit/algorithms/test_trim.py`：
   - config 默认值、注册表。
   - `_expand_pool`：pool 单调增大（未 break 时）、返回构型去重、psi 形状正确。
   - `_local_trim`：存活数 ≤ 输入、≤ `num_groups × keep_count`、构型唯一。
   - 使用小 Hubbard 模型（无外部依赖）。
3. 集成测试 `tests/integration/test_trim.py`（H2/STO-3G，openfermion）：
   - 跑 1-2 cycle（`max_cycles=1`），Krylov 能量有限且合理（> -2.0，< 0）。
   - target_psi 归一化正确（`|target[max_idx]| == 1`）。
4. 对照：同一 model 下 trim 的 global Krylov 能量应与 haar 同量级。

## 12. 附带发现

`haar.py:109-110` `_DynamicLanczos._extend` 传复数 1D `psi_weight` 给期望 `[batch,2]`
real/imag 的 `find_topk_relative_configs`——疑似格式不匹配。trim.py 明确传 `[batch,2]` 规避。
是否顺修 haar 由后续决定（不在本 scope）。
