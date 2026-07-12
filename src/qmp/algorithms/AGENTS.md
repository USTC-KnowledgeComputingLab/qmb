# Algorithms 子系统

## Registry

- ``action_config_dict[name] = ConfigClass`` — 配置 dataclass 类型
- ``action_class_dict[name] = ImplClass`` — 实现类

每个算法模块在底部注册，上方通过 ``importlib.import_module`` 触发。

## Action config 约定

每个 action config 包含 ``model: SubConfigRef | None`` 和 ``network: SubConfigRef | None`` 字段。
``None`` 表示该 action 不需要对应子系统。

- Model 构造: ``qmp.models._build.build_model(ref)``
- Network 构造: ``model.create_network(ref.name, ref.params, *, rngs)``

## trim action (TrimCI x HAAR)

`trim.py` 复现 TrimCI（arXiv:2511.14734）的 expansion + 两级 trim 选态引擎，
接到 HAAR 的 NQS 拟合回环。一个 cycle：网络采样 → 多跳全局 top-K 扩展（跳间用
`H·psi` 传播权重，实现真正图扩散）→ 随机分块 Lanczos (local trim) → 全局 Lanczos
(global trim) → `|psi|^2` 目标 → `_local_optimize` 拟合回网络 → 更新 pool。

**与 haar 的关系**：从 `haar.py` import 复用 `_DynamicLanczos`、`_local_optimize`、
`_sample_from_pool`、`_merge_pools`、checkpoint helper、`_AVAILABLE_LOSSES`、
`KrylovBasisStrategy`。haar.py 不改动。核心区别：haar 用单个（可自适应扩基的）
Lanczos 直接得目标；trim 用显式 expansion + 分块/全局两级对角化 trim。

**三个 TrimCI 阶段函数**：
- `_expand_pool`：多跳全局扩展。每跳 `find_topk_relative_configs` 取 top-K（psi 传
  `[batch, 2]` real/imag，与 kernel 契约一致），跳间用 `apply_within_subspace` 的
  `H·psi` 一阶振幅作下一跳权重。每跳用 `_unique_configs` 去重（top-K 在小系统会返回
  零填充行）。
- `_local_trim`：`jax.random.permutation` 随机分 `num_groups` 组，每组小 Lanczos
  取最低 Ritz，按 `|c|` 留每组 top-`local_keep_count`，`_merge_pools` 合并去重。
  随机分块是有意为之——廉价、无偏、保多样性；准确系数交给 global trim。
- global trim：主循环内直接用 `_DynamicLanczos`（`FIXED`），按 `|psi|` 选下轮 core。

**非目标**：COO 轨道优化、PT2 微扰、多随机 run ensemble、Davidson——均为后续工作。

