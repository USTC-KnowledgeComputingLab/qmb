# Spec: CLI 子系统与 Registry 架构

**日期**: 2026-07-13
**状态**: in-progress

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

顶层仅一个 `action` 节。`model` 和 `network` 下沉为 action params 内的 `SubConfigRef`，由每个 action 自行声明是否需要。`common` 节取消，种子/设备等字段由各 action config 自行决定是否包含。

### 2.2 CLI 入口 (`__main__.py`)

```python
import argparse, importlib, logging, typing
from dataclasses import dataclass, field
import tyro
from omegaconf import OmegaConf

@dataclass
class ActionCLI:
    name: str = "demo"
    params: dict[str, typing.Any] = field(default_factory=dict)

@dataclass
class ConfigCLI:
    action: ActionCLI = field(default_factory=lambda: ActionCLI(name="demo"))

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, remaining = pre.parse_known_args()
    defaults = ConfigCLI()
    if known.config:
        raw = OmegaConf.load(known.config)
        plain = OmegaConf.to_container(raw, resolve=True)
        assert isinstance(plain, dict)
        defaults.action = ActionCLI(**plain.get("action", {}))
    cli = tyro.cli(ConfigCLI, default=defaults, args=remaining)
    run_action(cli.action)
```

tyro 只负责 CLI override（`--action.name haar --action.params.sampling_count=2048`），YAML 加载由 omegaconf 完成（支持 `${}` 插值）。

### 2.3 优先级链

```
CLI args > YAML config > dataclass defaults
```

## 3. 子系统引用 (`SubConfigRef`)

```python
from dataclasses import dataclass, field
import typing

@dataclass
class SubConfigRef:
    """Reference from an action config to a subsystem (model/network)."""
    name: str
    params: dict[str, typing.Any] = field(default_factory=dict)
```

action config 包含 `model: SubConfigRef | None` 和 `network: SubConfigRef | None`。为 `None` 时表示该 action 不需要该子系统。

## 4. Registry 注册表

三个子系统各自维护一对注册表：Config + Impl。模块导入时自注册。

| 子系统 | Config 注册表 | Impl 注册表 | 所在包 |
|--------|-------------|-----------|--------|
| model | `model_config_dict` | `model_dict` | `qmp.models._model` |
| network | `network_config_dict` | `network_class_dict` | `qmp.networks._registry` |
| action | `action_config_dict` | `action_class_dict` | `qmp.algorithms._registry` |

各模块注册模式：

```python
# qmp.models.hubbard.py
from ._model import ModelProto, model_config_dict, model_dict

model_dict["hubbard"] = Model
model_config_dict["hubbard"] = ModelConfig
```

```python
# qmp.algorithms.haar.py
from ._registry import action_config_dict, action_class_dict

action_config_dict["haar"] = HaarConfig
action_class_dict["haar"] = Haar
```

```python
# qmp.networks.mlp.py
from ._registry import network_config_dict, network_class_dict

network_config_dict["mlp"] = MLPConfig
network_class_dict["mlp"] = MLP
```

## 5. 构造 Helper (`_build_from_ref`)

```python
import importlib, dacite, logging, typing

def build_from_ref(ref: SubConfigRef, subsystem: str, *,
                   config_dict: dict, impl_dict: dict) -> typing.Any:
    """从 SubConfigRef 构造子系统实例。"""
    if ref.name not in config_dict:
        importlib.import_module(f"qmp.{subsystem}.{ref.name}")
    cfg_cls = config_dict[ref.name]
    cfg = dacite.from_dict(cfg_cls, ref.params)
    impl_cls = impl_dict[ref.name]
    return impl_cls(cfg)
```

放置位置：`src/qmp/utility/_build.py`（utility 用于共享工具）。

## 6. Action 实现示例

```python
# qmp.algorithms.demo.py
from dataclasses import dataclass, field
from qmp.algorithms._registry import action_config_dict, action_class_dict
from qmp.models._model import model_config_dict, model_dict
from qmp.utility._build import build_from_ref

@dataclass
class DemoConfig:
    model: SubConfigRef | None = None
    network: SubConfigRef | None = None
    message: str = "Hello"

class Demo:
    def __init__(self, config: DemoConfig):
        self._model = build_from_ref(config.model, "models",
                                      config_dict=model_config_dict, impl_dict=model_dict)
        self._network = build_from_ref(config.network, "networks",
                                        config_dict=network_config_dict, impl_dict=network_class_dict)

    def run(self) -> None:
        ...

action_config_dict["demo"] = DemoConfig
action_class_dict["demo"] = Demo
```

## 7. CLI dispatch (`run_action`)

```python
def run_action(cli: ActionCLI) -> None:
    importlib.import_module(f"qmp.algorithms.{cli.name}")
    cfg_cls = action_config_dict[cli.name]
    cfg = dacite.from_dict(cfg_cls, cli.params)
    impl_cls = action_class_dict[cli.name]
    instance = impl_cls(cfg)
    instance.run()
```

## 8. 与旧版 CLI (Hydra) 的区别

| 维度 | 旧 Hydra | 新版 |
|------|---------|------|
| 配置 schema | 隐式 (YAML 写什么就什么) | 显式 (action config dataclass) |
| model/network 位置 | 顶层 | 下沉到 action params |
| 构造责任 | context.py 统一构造 | action 内部通过 `build_from_ref` |
| 依赖 | hydra-core, omegaconf, dacite | tyro, omegaconf, dacite |
| 类型安全 | 无 | dataclass + dacite 运行时校验 |

## 9. 非目标

- tyro subcommand 类型安全 CLI 体验：主要场景是 YAML 配置
- 多节点 shard_map 集成
- action 间组合/pipeline (未来扩展)

## 10. 需要更新的 AGENTS.md

- `src/qmp/AGENTS.md` (root)：添加 CLI 架构说明
- `src/qmp/algorithms/AGENTS.md` (new)：registry + SubConfigRef 模式
- `src/qmp/models/AGENTS.md`：更新 `model_config_dict` 注册说明
- `src/qmp/utility/AGENTS.md` (new)：`build_from_ref` helper 说明
