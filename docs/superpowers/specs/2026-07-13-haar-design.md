# Spec: HAAR 算法迁移

**日期**: 2026-07-13
**状态**: in-progress

## 1. 目标

将 HAAR（Hybrid Adaptive Antisymmetric Representation）算法从旧 PyTorch 代码迁移到新 JAX/Flax 栈，适配新的 CLI、Registry、ModelProto/NetworkProto 接口。

## 2. 算法概述

两阶段交替优化：

1. **Krylov 虚时演化**：在有限构型基上执行 Lanczos 对角化，可选四种构型基扩展策略
2. **Local Optimization**：以 Krylov 结果为目标，用梯度优化器微调网络参数

### 2.1 Krylov + 构型基扩展

在构型基 `{configs}` 上执行 Lanczos 对角化 `H_{ij} = ⟨cᵢ|H|cⱼ⟩` 得到基态/激发态。四种扩展策略：

| `KrylovBasisStrategy` | 行为 |
|------|------|
| `FIXED` | 不扩展。给定基上纯 Lanczos |
| `PRECOMPUTE` | Lanczos 前：`max_steps` 次 H·ψ 幂迭代，在最高概率构型上扩展基，再 Lanczos |
| `POSTCOMPUTE` | Lanczos 后：所有 Krylov 向量的概率密度之和作为权重，扩展一次，再跑第二次 Lanczos |
| `ADAPTIVE` | 每步 Lanczos 对角化后用最新 Krylov 向量扩展基，Lanczos 从头重启。逐步趋近 |

**随机向量注入**：`krylov_random_period < krylov_max_steps` 时，每 `krylov_random_period` 步（以及 `norm < krylov_stop_norm` 时）注入正交随机向量防数值退化。默认 `krylov_random_period = max_steps - 1`（Lanczos 最后一步触发一次）。

### 2.2 Local Optimization

Krylov 给出目标波函数 `ψ_target`。网络参数 θ 通过梯度下降最小化 `loss(ψ_net, ψ_target)`。NaN/inf 检测后回退状态。

## 3. 配置 dataclass

```python
from enum import Enum

class KrylovBasisStrategy(Enum):
    FIXED = "fixed"
    PRECOMPUTE = "precompute"
    POSTCOMPUTE = "postcompute"
    ADAPTIVE = "adaptive"

@dataclass
class HaarConfig:
    model: SubConfigRef | None = None
    network: SubConfigRef | None = None

    # sampling
    sampling_count_from_network: int = 1024
    sampling_count_from_pool: int = 1024

    # krylov
    krylov_max_steps: int = 32
    krylov_stop_norm: float = 1e-8
    krylov_random_period: int = 31        # default = max_steps - 1
    krylov_state_count: int = 1
    basis_extend_count: int = 64
    basis_strategy: KrylovBasisStrategy = KrylovBasisStrategy.ADAPTIVE

    # local optimization
    loss_name: str = "sum_filtered_angle_scaled_log"
    local_max_steps: int = 10000
    local_stop_loss: float = 1e-8
    local_log_psi_count: int = 30

    # checkpoint
    checkpoint_path: str | None = None
    checkpoint_interval: int = 1
```

**随机注入逻辑**：仅在 `krylov_random_period < krylov_max_steps` 时生效。`step % krylov_random_period == 0` 或 `norm < krylov_stop_norm` 时注入。

## 4. 状态 dict

```python
state = {
    "haar": {
        "global": 0,           # cycle 计数
        "local": 0,            # 累计 local step 计数
        "pool": (configs, psi, counts),   # jax arrays
        "excited": [          # 每项: (energy, configs, psi)
            (E0, configs, psi),
            ...
        ],
    },
    "network": nnx_params,     # pickle 序列化
    "optimizer": optax_state,  # pickle 序列化
}
```

- `pool`: `(configs: [M, Q] uint8, psi: [M] complex128, counts: [M] int32)`
- `excited`: 列表，每项 `(energy: float, configs: [M, Q] uint8, psi: [M] complex128)`。按 `krylov_state_count` 控制数量
- 网络参数和优化器状态在顶层 key，加载时直接恢复

## 5. 主循环伪代码

```python
def run(self) -> None:
    state = load_checkpoint() or init_state()

    while True:
        # 1. 采样
        configs_net, psi_net, _ = network.generate_unique(sampling_count_from_network, key=key)
        configs_pool, psi_pool = sample_from_pool(state["haar"]["pool"])
        configs, psi = merge_and_dedup(configs_net, psi_net, configs_pool, psi_pool)

        # 2. Krylov
        lanczos = DynamicLanczos(model, configs, psi, ...)
        for results in lanczos.run():
            E, configs, psi = results[0]
            log energy, error
        state["haar"]["excited"] = results   # 最新一轮结果

        # 3. Target construction
        psi_target = sum over lanczos states (probability-weighted)

        # 4. Local optimization
        optax optimizer loop: minimize loss(ψ_net, ψ_target)
        NaN/inf detection → restore backup

        # 5. Update pool
        state["haar"]["pool"] = (configs, psi, counts)
        state["haar"]["global"] += 1
        save checkpoint if interval reached
```

## 6. DynamicLanczos

内部类/独立模块。核心算法：

```
输入: model, configs, psi, max_steps, stop_norm, random_period, extend_count, basis_strategy, state_count

_run() 生成器:
    v[0] = psi / norm(psi)
    w = H·v[0]; alpha[0] = ⟨v[0]|w⟩; yield (alpha, beta, v)
    w -= alpha[0]·v[0]
    loop:
        norm_w = |w|
        if norm_w < stop_norm or step % random_period == 0:
            inject random vector (orthogonalized)
        else:
            v[k+1] = w / norm_w
            beta[k] = norm_w
        w = H·v[-1]; alpha[k+1] = ⟨v[-1]|w⟩; yield (alpha, beta, v)
        w -= alpha[k+1]·v[-1] - beta[k]·v[-2]
        reorthogonalize w against all v

_eigh(): jnp.linalg.eigh_tridiagonal(alpha, beta) → eigenvals, eigenvecs

_extend(psi_weight): model.find_topk_relative_configs(configs, psi_weight, extend_count, configs)

run() 外层: 根据 basis_strategy 调度 _run() 和 _extend()
```

**与旧代码差异：**
- 使用 `jnp.linalg.eigh` 对角化 Krylov 子空间矩阵（JAX 无 `eigh_tridiagonal`）。先从 `alpha/beta` 构建三对角矩阵 `[max_steps, max_steps]`，再 `eigh`。Krylov 维度 ≤ 256，O(N³) 成本可忽略。
- 生成器模式改纯函数 + Python for loop

## 7. Loss 函数

从旧 `losses.py` 迁移。首批只迁移 `sum_filtered_angle_scaled_log`（最常用的）。loss 函数签名：

```python
def loss_fn(psi_net: jax.Array, psi_target: jax.Array) -> jax.Array:
    """psi_net, psi_target: [batch] complex128 → scalar loss"""
```

## 8. Checkpoint

```python
import pickle, pathlib

def save_checkpoint(state: dict, path: pathlib.Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(state, f)

def load_checkpoint(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
```

## 9. 注册

```python
# haar.py 底部
from qmp.algorithms._registry import action_config_dict, action_class_dict
action_config_dict["haar"] = HaarConfig
action_class_dict["haar"] = Haar
```

## 10. 依赖

- `jax`, `jax.numpy`, `jax.lax`
- `optax` — optimizer
- `flax.nnx` — network (rngs)
- `dacite` — config 构造
- 无: `scipy`, `torch`, `omegaconf`

## 11. 非目标

- TensorBoard / wandb 日志：第一阶段用 logging
- Multi-GPU shard_map：单卡优先
- Mixed-precision bf16/fp32：全 float64

## 12. 需要验证的点

1. `jnp.linalg.eigh` 是否支持三对角？标准 API 是 `jnp.linalg.eigh_tridiagonal(a, b)` — **需确认 JAX 是否有此函数**。若无则用 `jnp.linalg.eigh`（O(N³) vs O(N²) 但对 Krylov 维度 ~100 影响可忽略）
2. VMC 采样 → 能量期望值闭环验证
3. 与旧代码对拍（小系统如 H₂ Hubbard，数值一致性）
