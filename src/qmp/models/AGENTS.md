# Models 子系统设计

## 概述

Models 子系统是**物理问题的定义层**：把某个具体的量子多体系统（分子、格点模型）翻译成统一的内部表示，作为算法层与哈密顿量层之间的桥梁。models 自身不做数值计算，而是**编排**其他子系统。

每个 model 承担三件事：

1. **构造哈密顿量**：把系统特有的输入格式转成统一的 `{term: complex_coefficient}` 字典，交给 `FermiHamiltonian`
2. **暴露统一算符接口**：转发 `FermiHamiltonian` 的四个核心操作 + `show_config` + 元数据
3. **注册可用网络**：`network_dict` 把物理系统与兼容的 ansatz 配对（本轮留空，待 networks 子系统迁移）

## 目录结构

```
src/qmp/models/
├── __init__.py       # 模块 docstring
├── _model.py         # ModelProto 协议 + model_dict 注册表
├── hubbard.py        # 二维格点 Hubbard 模型 (程序化生成，无外部依赖)
├── fcidump.py        # FCIDUMP 文件 (openfermion 解析 + pickle 缓存)
└── openfermion.py    # OpenFermion MolecularData 文件
```

## 设计原则

### 薄封装：model 是编排者，不是计算者

每个 model 只负责把自身输入翻译为 `{term: coef}` 字典，其余全部委托给 `FermiHamiltonian`。四个算符方法 (`compute_diagonal_within_subspace`, `apply_within_subspace`, `find_all_relative_configs`, `find_topk_relative_configs`) 都是对内部 `self.hamiltonian` 的直接转发。

### 转发样板刻意不抽基类

三个 model 的四个算符转发方法 + `show_config` + `_show_config_site` 几乎逐字重复。这是**刻意保留**的：保证未来各 model 独立演化的自由度（不同 model 可能需要不同的构型渲染、不同的默认参数）。不要为消除重复而引入共享基类。

### 插件式注册表

- 全局 `model_dict[name] = ModelClass`：配置驱动的 CLI 按名字动态查找 model
- 全局 `model_config_dict[name] = ModelConfigClass`：CLI 层 `dacite.from_dict` 按名字查找 config dataclass 类型
- 每个 model 的 `network_dict`：Model 支持哪些 network config 类型（`{"mlp": MLPConfig, ...}`）
- 每个 model 的 `create_network(name, params)`：model 调用 dacite + 注入自身属性（`sites`, `n_electrons` 等）构造 network 实例

**network 构造设计**：network 不设全局注册表。由 model 内聚——model 知道自己的系统尺寸，故由其负责构造 network。network config dataclass 是纯数据（`hidden`, `activation`），不含 `create` 方法。

model 通过在模块底部执行 `model_dict[name] = Model` 和 `model_config_dict[name] = ModelConfig` 完成自注册；上层通过 `importlib.import_module` 动态导入触发注册。

## ModelProto 接口

| 方法/属性 | 语义 |
|-----------|------|
| `compute_diagonal_within_subspace(configs)` | 对角元 H[i,i] |
| `apply_within_subspace(configs_i, psi_i, configs_j, *, direction=0)` | H·ψ 投影到目标子空间 |
| `find_all_relative_configs(configs_i, psi_i, configs_exclude=None, *, hash_capacity)` | 枚举全部新构型 + 去重 |
| `find_topk_relative_configs(configs_i, psi_i, count_selected, configs_exclude=None)` | Top-K 最重要新构型 |
| `show_config(config) -> str` | 位编码构型渲染为可读字符串 |
| `ref_energy: float` | 参考能量 |
| `network_dict: ClassVar[dict]` | 兼容网络注册表 (本轮空) |

`configs_exclude` 为 `None` 时，转发层填入空数组 `jnp.zeros((0, n_qubytes), uint8)`；`FermiHamiltonian` 的这两个方法要求该参数非可选。

## 三个 model

### hubbard

二维格点 Hubbard 模型，程序化生成哈密顿量，**无外部依赖**。site 索引 `(i + j*m)*2 + o`，`o=0/1` 分别为自旋上/下。项：最近邻 hopping (`-t`)、on-site 相互作用 (`u`)、化学势 (`-mu`)。`electron_number` 默认半填充 `m*n`。不含自旋量子数（网络阶段取一半）。

### fcidump

解析 FCIDUMP 文件（支持 `.gz`）。头部读 `NORB`/`NELEC`/`MS2`；单/双电子积分展开为自旋轨道，经 `openfermion.InteractionOperator → get_fermion_operator → normal_ordered` 得字典。元数据 `n_spins = ms2`。

- **`ref_energy` 只取自 config，默认 0.0**，不读 `FCIDUMP.yaml`（与旧代码不同）
- **缓存**：见下节

### openfermion

加载 OpenFermion `MolecularData` 文件。元数据 `n_qubits`, `n_electrons`, **`n_spins = multiplicity − 1`**（multiplicity = 2S+1；旧代码硬编码 S_z=0 是 bug，此处用真实自旋修复），`ref_energy = fci_energy`。哈密顿量经 `get_fermion_operator(get_molecular_hamiltonian()).terms`。

## 自旋量子数约定

`n_spins` 统一表示 N↑ − N↓ = 2·S_z：
- FCIDUMP: `n_spins = ms2`
- MolecularData: `n_spins = multiplicity − 1`

网络阶段据此拆分 `spin_up = (n_electrons + n_spins) // 2`, `spin_down = (n_electrons − n_spins) // 2`。

## 缓存

fcidump 解析结果缓存于 `~/.cache/qmp/models/fcidump/{sha256}-v1.pkl`：
- key = 文件内容的 sha256 + 版本号
- 内容 = 哈密顿量字典 (pickle，因 term key 是变长 tuple)
- 缓存根目录由模块级 `_cache_dir()` 提供，便于测试用 monkeypatch 重定向

缓存布局与 hamiltonian 子系统的 CUDA 编译产物 (`~/.cache/qmp/hamiltonian/fermi/`) 按子系统分命名空间隔离。

## 依赖

- `openfermion`：fcidump + openfermion 两个 model 需要（含 numpy/scipy）
- `jax`：数值数组类型与转发
- `platformdirs`：缓存目录定位

hubbard 不依赖 openfermion。

## device 参数

各 `ModelConfig` 带 `devices: list[str] = ["localhost:cpu:0"]`，透传给 `FermiHamiltonian`。models 不解析 device，仅转发。
