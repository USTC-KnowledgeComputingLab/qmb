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
