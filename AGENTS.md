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

- **Hamiltonian**：参见 `src/qmp/hamiltonian/AGENTS.md`，包含 C++ 内核/接口架构、CPU/CUDA 差异、设备归属、JIT 编译等设计细节。

## 设计原则

### 纯函数优于副作用

输出通过返回值传递，不通过可变的引用参数。`const` 输入 + 纯输出 = 更安全、更易测试。

反例（已修正）：`sort_configs(configs, /*out*/ sort_idx)`  
正例：`auto [sorted, sort_idx] = sort_configs(configs)`

### 文档是交付物

每个 spec、plan、AGENTS.md 必须在代码稳定后统一检查并更新。过时的文档和过时的测试断言一样，都是 bug。每次提交前确认：
- spec 中的 design decisions 表与代码一致
- plan 的 status 标记（approved / implemented / completed）正确
- AGENTS.md 中引用的函数名、参数顺序、模块名与代码同步

### 参考旧代码

`old/` 中的 main 分支代码是历经迭代验证的参考实现。新代码不必逐行复刻，但旧代码中的关键模式（CUDAGuard + stream、thread block 维度设计、thrust 与 kernel 同流等）是经过正确性验证的设计选择，不应随意偏离。

### 设计先于实现

任何非平凡的改动先产出 spec（要做什么、决策表）和 plan（分步实施），再动手写代码。跳过这一步会导致反复返工和在未确认方向上浪费时间。
