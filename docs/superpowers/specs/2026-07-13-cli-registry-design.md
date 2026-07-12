# Spec: CLI 子系统与 Registry 架构

**日期**: 2026-07-13
**状态**: implemented（`build_from_ref` 已重构为 `build_model` + `model.create_network`；本文档反映当前实现）

## 1. 目标

定义 qmp-kit 的 CLI 入口、YAML 配置结构、以及 model/network/action 三个子系统的注册与构造机制。

## 2. CLI 配置结构

### 2.1 YAML 配置

```yaml
action:
  name: haar
  params:
    seed: 2333
    device: auto
    sampling_count: 1024
    model:
      name: fcidump
      params:
        model_path: /tmp/test.fcidump
    network:
      name: mlp
      params:
        hidden: [512, 512]
```

顶层仅一个 `action` 节。`model` 和 `network` 下沉为 action params 内的 `SubConfigRef`。

### 2.2 CLI 入口 (`__main__.py`)

tyro 只负责 CLI override，YAML 加载由 omegaconf 完成（支持 `${}` 插值）。优先级: `CLI args > YAML > dataclass defaults`。

## 3. 子系统引用 (`SubConfigRef`)

```python
@dataclass
class SubConfigRef:
    name: str
    params: dict[str, typing.Any] = field(default_factory=dict)
```

action config 包含 `model: SubConfigRef | None` 和 `network: SubConfigRef | None`。

## 4. Registry 注册表

model 和 action 各自维护双注册表（Config + Impl）。**network 不设全局注册表——由 model 内聚。**

| 子系统 | Config 注册表 | Impl 注册表 |
|--------|-------------|-----------|
| model | `model_config_dict[name] = ConfigClass` | `model_dict[name] = ImplClass` |
| network | `model.network_dict[name] = ConfigClass` (per-model) | model 内部管理 |
| action | `action_config_dict[name] = ConfigClass` | `action_class_dict[name] = ImplClass` |

各模块自注册：

```python
# models/hubbard.py
model_dict["hubbard"] = Model
model_config_dict["hubbard"] = ModelConfig

# algorithms/haar.py
action_config_dict["haar"] = HaarConfig
action_class_dict["haar"] = Haar
```

## 5. 构造机制

### 5.1 Model: `build_model`

```python
def build_model(ref: SubConfigRef | None) -> typing.Any:
    if ref is None:
        return None
    if ref.name not in model_config_dict:
        try:
            importlib.import_module(f"qmp.models.{ref.name}")
        except ModuleNotFoundError:
            raise KeyError(f"Unknown model: {ref.name!r}") from None
    cfg_cls = model_config_dict[ref.name]
    cfg = dacite.from_dict(cfg_cls, ref.params)
    impl_cls = model_dict[ref.name]
    return impl_cls(cfg)
```

放置位置：`src/qmp/models/_build.py`。

### 5.2 Network: `model.create_network`

network 无全局注册表。每个 model 类拥有 `network_dict`（支持哪些 network config 类型）；network config dataclass 含 `create(self, model, *, rngs)` 工厂方法。model 的 `create_network` 是薄包装：

```python
# models/hubbard.py
def create_network(self, name: str, params: dict, *, rngs: nnx.Rngs) -> NetworkProto:
    cfg_cls = self.network_dict[name]
    cfg = dacite.from_dict(cfg_cls, params)
    return cfg.create(self, rngs=rngs)
```

network config 是纯数据 + 工厂：

```python
@dataclass
class MlpUpDownConfig:
    hidden_size: list[int] = field(default_factory=lambda: [512])
    ordering: int = 1

    def create(self, model: Model, *, rngs: nnx.Rngs) -> NetworkProto:
        return WaveFunctionElectron(
            double_sites=model.n_qubits,
            spin_up=model.electron_number // 2,
            hidden_size=tuple(self.hidden_size),
            ordering=self.ordering,
            rngs=rngs,
        )
```

action 调用: `network = model.create_network(ref.name, ref.params, rngs=nnx.Rngs(42))`。

## 6. Action 实现示例

```python
class Demo:
    def __init__(self, config: DemoConfig):
        self._model = build_model(config.model)
        self._network = None
        if config.network is not None and self._model is not None:
            self._network = self._model.create_network(
                config.network.name, config.network.params, rngs=nnx.Rngs(42)
            )
```

## 7. CLI dispatch

```python
def main():
    # ... YAML load + tyro override ...
    importlib.import_module(f"qmp.algorithms.{cli.action.name}")
    cfg_cls = action_config_dict[cli.action.name]
    cfg = dacite.from_dict(cfg_cls, cli.action.params)
    impl_cls = action_class_dict[cli.action.name]
    instance = impl_cls(cfg)
    instance.run()
```

## 8. 与旧版 CLI (Hydra) 的区别

| 维度 | 旧 Hydra | 新版 |
|------|---------|------|
| 配置 schema | 隐式 | dataclass |
| model/network 位置 | 顶层 | 下沉到 action params |
| model 构造 | context 统一构造 | `build_model(ref)` (全局 registry) |
| network 构造 | `config.create(model)` 在 network 侧 | `model.create_network(name, params)` 在 model 侧 |
| 依赖 | hydra-core, omegaconf, dacite | tyro, omegaconf, dacite |

## 9. 需要更新的 AGENTS.md

- `AGENTS.md` (root)：CLI 架构说明
- `src/qmp/algorithms/AGENTS.md`：registry + SubConfigRef 模式
- `src/qmp/models/AGENTS.md`：`model_config_dict` + `model.create_network` 约定
