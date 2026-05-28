# qmp-kit 开发指南

## 快速开始

```bash
uv sync
uv run pytest
```

## 依赖管理

所有依赖使用 `uv` 管理：

```bash
uv sync                    # 安装所有依赖 + 开发依赖
uv sync --no-dev           # 仅安装运行时依赖
uv add <package>           # 添加运行时依赖
uv add --dev <package>     # 添加开发依赖
uv lock                    # 手动修改 pyproject.toml 后更新 uv.lock
```

## 代码质量

```bash
uv run ruff format .           # 自动格式化
uv run ruff check .            # 静态检查
uv run ty check src/qmp tests  # 类型检查
```

所有工具配置均在 `pyproject.toml` 中。

## 运行测试

```bash
uv run pytest                       # 全部测试
uv run pytest tests/test_file.py    # 指定测试文件
uv run pytest -k "pattern"          # 按名称匹配测试
```

## 代码规范

- **字符串字面量**：使用双引号（`"`）用于字符串，单引号（`'`）仅用于类字符场景
- **行长度**：120 字符（ruff 配置）
- **导入顺序**：按 isort 规则排序（标准库 → 第三方 → qmp）
- **类型注解**：所有函数签名和非平凡的 method 都必须有类型注解（ty 强制）
- **文档字符串**：尽量使用 NumPy 风格

## 项目结构

```
src/qmp/          # 源码（src layout）
tests/            # 测试
```

## 构建系统

- **构建后端**：hatchling
- **版本管理**：hatch-vcs（从 git tag 自动生成）
- **包布局**：src layout（`src/qmp/`）

## 子系统设计

- **Hamiltonian**：参见 `src/qmp/hamiltonian/AGENTS.md`，包含 C++ 内核/接口架构、设备归属、JIT 编译等设计细节。
