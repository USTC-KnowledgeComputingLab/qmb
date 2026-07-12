# AGENTS.md

本文档面向后续开发者（人或 AI Agent），说明 qmp-kit 的设计思路、工程约定及开发实践。

## 技术栈

| 领域 | 技术 |
|------|------|
| 数值计算 | `jax` + `jaxlib` |
| 神经网络 | `flax` (nnx API) |
| CUDA 扩展 | `jax.ffi` (XLA custom call) + nvcc + cuCollections |
| 量子化学接口 | `openfermion`, `pyscf` |
| Lint/Format | `ruff` |
| 类型检查 | `ty` (Astral 出品) |
| 测试 | `pytest` |
| 构建 | `uv` + `hatchling` + `hatch-vcs` |
| Python | >= 3.13 |

## 代码结构

```
src/qmp/                # 源码 (src layout)
├── __init__.py
├── _version.py          # hatch-vcs 自动生成
├── hamiltonian/         # Hamiltonian 子系统
│   └── fermi_hamiltonian/  # Fermi Hamiltonian CUDA kernel + Python wrapper
├── networks/            # MLP / Transformers / MPS (Flax)
├── algorithms/          # Action registry + demo
├── models/              # FCIDUMP / Hubbard / Ising / PySCF / OpenFermion
├── plugins/             # 第三方框架接口
└── utility/             # bitspack, losses, context, optimizer

tests/                   # 测试 (镜像 src/ 结构)
docs/superpowers/        # 设计 spec + plan
old/                     # 旧 main 分支参考代码 (如存在)
```

## CLI 架构

`qmp --config config.yaml` 入口在 `__main__.py`。YAML 仅含 `action` 顶层节；model/network 作为 `SubConfigRef(name, params)` 嵌入 action params。

model 和 action 维护双注册表（`xxx_config_dict` + `xxx_class_dict`），模块导入时自注册。action 内部通过 `qmp.models._build.build_model` 从 `SubConfigRef` 构造 model；network 通过 `model.create_network(name, params, *, rngs)` 构造（network 不设全局注册表，由 model 内聚）。

## 类型注解约定

- 使用 `from __future__ import annotations` 在所有文件
- `X | None` 而非 `Optional[X]`
- `list[X]` 而非 `List[X]`
- 禁止 `from typing import Any` 后使用 raw `Any`——始终 `typing.Any`
- 局部变量使用描述性全小写名称 (`batch_size`, `term_count`, `count_selected`, `n_qubytes`)，禁止单字母大写变量名 (`B`, `T`, `Q`, `K` 等)

## 字符串与格式约定

- **字符串**: 双引号 (`"`)，单引号仅用于类字符场景 (`"'}` 这类)
- **行长度**: 120 (ruff 配置)
- **导入顺序**: isort 规则 (标准库 → 第三方 → `qmp`)

## 设计原则

### 设计先于实现

非平凡的改动先产出 spec (NOTE.md 或 docs/superpowers/specs/) 和 plan，再动手写代码。跳过这一步会导致反复返工。

### 文档是交付物

每个 spec、plan、AGENTS.md 必须在代码稳定后统一检查并更新。过时的文档和过时的测试断言一样，都是 bug。

### 纯函数优于副作用

输出通过返回值传递，不通过可变引用参数。`const` 输入 + 纯输出 = 更安全、更易测试。

### 零抑制原则

不堆 `# noqa`，不堆 `# ty: ignore`。代码本身应符合 lint 和类型检查规则。仅在以下情况允许 suppress：
- 第三方库无类型 stubs（如 `jaxlib.get_include_dir`）
- try/except 中的延迟 import（可选依赖）

每条 suppress 必须附带注释说明理由。

### 性能选择需有可解释的理由

不是"这样更快"，而是"这样更快，因为..."。关键设计决策必须记录原因（参考 `src/qmp/hamiltonian/AGENTS.md` 中的性能说明）。

### 多节点多卡是首要目标

所有核心操作从设计之初就考虑 `shard_map` 兼容性：
- configs 按 batch 维度分片，Hamiltonian 复制
- FFI kernel 通过 `shard_map` 分发，避免隐式 all-gather
- 全局同步通过 JAX collectives (`psum`, `all_gather`)，不手写 MPI

### 全局启用 float64

`src/qmp/__init__.py` 中 `jax.config.update("jax_enable_x64", True)`。log-amplitude、phase、Hamiltonian 系数全部以 float64 累加。网络参数用 `param_dtype=jnp.float64` 显式声明。

### networks 子系统 (Flax nnx)

- 自回归神经量子态 (ANQS)：`__call__(configs)->psi`、`generate`、`generate_unique`，契约见 `networks/_protocol.py`。
- `generate` / `generate_unique` 显式接收 PRNG `key`（随机性无内部状态）。
- 生成循环用 unrolled Python for（sites 静态已知，逐 site 展开；所有静态形状函数显式 jit，`generate*` 因动态形状不整体 jit）；共享逻辑（mask、归一化、Gumbel top-K 束搜索）在 `networks/_autoregressive.py`。
- Gumbel 条件截断用有限 sentinel + `jnp.where` 护栏防 `inf - inf = NaN`。

## 开发命令

```bash
uv sync                        # 安装依赖
uv run ruff format .           # 格式化
uv run ruff check .            # lint
uv run ty check src tests      # 类型检查
uv run pytest -v               # 测试
```

## License

AGPL-3.0-only
