# CLI + Registry 架构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **实现后演进（务必先读）**：本 plan 记录初始实现步骤，其中的 `build_from_ref`（通用 subsystem helper）与 `networks/_registry.py`（全局 network 注册表）**后续已被重构替代**，当前代码与 `docs/superpowers/specs/2026-07-13-cli-registry-design.md` 为准：
> - `build_from_ref(...)` → `qmp.models._build.build_model(ref)`（仅构造 model）
> - network 无全局注册表；由 `model.create_network(name, params, *, rngs)` 内聚构造（`networks/_registry.py` 已删除）
> - `_build.py` 从 `utility/` 迁至 `models/`
> 下文 Task 步骤按历史原样保留，不逐字回改。

**Goal:** 实现 CLI 与 Registry 架构：YAML 配置只有 `action` 顶层，action config 通过 `SubConfigRef` 嵌入 model/network，共享 `build_from_ref` helper 构造实例，三个子系统统一双注册表模式。

**Architecture:** `YAML → omegaconf → ConfigCLI(action) → tyro override → dispatch action → action 内部通过 build_from_ref 从 SubConfigRef 构造 model/network`

**Tech Stack:** tyro, omegaconf, dacite, jax

---

## File Structure

| 文件 | 职责 |
|------|------|
| `src/qmp/utility/_build.py` (new) | `SubConfigRef` dataclass + `build_from_ref` helper |
| `src/qmp/utility/__init__.py` (new) | utility 包 init |
| `src/qmp/networks/_registry.py` (new) | `network_config_dict` + `network_class_dict` + 空占位 |
| `src/qmp/networks/__init__.py` (new) | networks 包 init |
| `src/qmp/algorithms/_registry.py` (rewrite) | `action_config_dict` + `action_class_dict` |
| `src/qmp/algorithms/*.py` (rewrite) | actions (haar, trim) 使用 SubConfigRef |
| `src/qmp/__main__.py` (rewrite) | 简化为 action dispatch |

---

### Task 1: `SubConfigRef` + `build_from_ref` helper

**Files:**
- Create: `src/qmp/utility/__init__.py`
- Create: `src/qmp/utility/_build.py`

- [ ] **Step 1: Write utility package init**

```python
"""Utility functions shared across subsystems."""
```

File: `src/qmp/utility/__init__.py`

- [ ] **Step 2: Write `_build.py`**

```python
"""Subsystem reference and construction helper.

``SubConfigRef`` is an (name, params) pair that action configs embed to
declare which model/network they need. ``build_from_ref`` uses the
subsystem registry pairs to construct instances from these refs.
"""

from __future__ import annotations

import importlib
import logging
import typing
from dataclasses import dataclass, field

import dacite

logger = logging.getLogger(__name__)


@dataclass
class SubConfigRef:
    """Reference from an action config to a subsystem (model/network)."""

    name: str
    params: dict[str, typing.Any] = field(default_factory=dict)


def build_from_ref(
    ref: SubConfigRef | None,
    subsystem: str,
    *,
    config_dict: dict[str, type[object]],
    impl_dict: dict[str, type[object]],
) -> typing.Any:
    """Construct a subsystem instance from a SubConfigRef.

    If ``ref`` is ``None``, returns ``None`` (action doesn't need this subsystem).
    Otherwise, lazy-imports the module, deserialises ``ref.params`` into the
    registered config dataclass, and calls the implementation class with it.

    Parameters
    ----------
    ref : SubConfigRef | None
        The reference from the action config.
    subsystem : str
        Package name for lazy import (e.g. ``"models"``, ``"networks"``).
    config_dict : dict
        Registry mapping name → config dataclass type.
    impl_dict : dict
        Registry mapping name → implementation class type.

    Returns
    -------
    typing.Any
        The constructed subsystem instance, or ``None`` if ``ref`` is ``None``.
    """
    if ref is None:
        return None
    if ref.name not in config_dict:
        importlib.import_module(f"qmp.{subsystem}.{ref.name}")
    cfg_cls = config_dict[ref.name]
    cfg = dacite.from_dict(cfg_cls, ref.params)
    impl_cls = impl_dict[ref.name]
    return impl_cls(cfg)
```

File: `src/qmp/utility/_build.py`

- [ ] **Step 3: Run ruff + ty + pytest**

```bash
uv run ruff format . && uv run ruff check src/qmp/utility/ && uv run ty check src/qmp/utility/
```

Expected: all clean.

- [ ] **Step 4: Commit**

```bash
git add src/qmp/utility/
git commit -m "feat: add SubConfigRef + build_from_ref utility helper"
```

---

### Task 2: Networks registry skeleton

**Files:**
- Create: `src/qmp/networks/__init__.py`
- Create: `src/qmp/networks/_registry.py`

- [ ] **Step 1: Write networks package init**

```python
"""Neural network ansatze for quantum many-body wavefunctions."""
```

File: `src/qmp/networks/__init__.py`

- [ ] **Step 2: Write `_registry.py`**

```python
"""Network registry.

``network_config_dict`` maps network names to their config dataclass type.
``network_class_dict`` maps network names to their implementation class.
Networks register themselves at import time.
"""

from __future__ import annotations

network_config_dict: dict[str, type[object]] = {}
network_class_dict: dict[str, type[object]] = {}
```

File: `src/qmp/networks/_registry.py`

- [ ] **Step 3: Run ruff + ty**

```bash
uv run ruff format . && uv run ruff check src/qmp/networks/ && uv run ty check src/qmp/networks/
```

- [ ] **Step 4: Commit**

```bash
git add src/qmp/networks/
git commit -m "feat: add networks registry skeleton"
```

---

### Task 3: Rewrite algorithms registry

**Files:**
- Rewrite: `src/qmp/algorithms/_registry.py`

- [ ] **Step 1: Rewrite `_registry.py`**

Current file has `action_dict` which maps name → ConfigClass but no ImplClass mapping. Replace with dual registry:

```python
"""Algorithm registry.

``action_config_dict`` maps action names to their config dataclass type.
``action_class_dict`` maps action names to their implementation class.
Algorithms register themselves at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qmp.models._model import ModelProto

from qmp.models._model import model_dict  # noqa: F401 — re-export for convenience


@runtime_checkable
class ActionProto(Protocol):
    """Uniform interface for all algorithms."""

    def run(self, model: ModelProto[object]) -> None:
        """Execute the algorithm."""
        ...


action_config_dict: dict[str, type[object]] = {}
action_class_dict: dict[str, type[object]] = {}
```

File: `src/qmp/algorithms/_registry.py`

- [ ] **Step 2: Commit**

```bash
git add src/qmp/algorithms/_registry.py
git commit -m "refactor: split action_dict into action_config_dict + action_class_dict"
```

---

### Task 4: Verify action registry with real algorithms

**Files:**
- Verify existing: `src/qmp/algorithms/haar.py`, `src/qmp/algorithms/trim.py`

The demo action has been removed. Registry validation is covered by per-algorithm
tests (``test_haar.py`` / ``test_trim.py``), each checking that ``action_config_dict``
and ``action_class_dict`` contain the expected keys.

- [x] **Step 1: Confirm algorithms registered**

Existing tests verify: ``action_config_dict["haar"] is HaarConfig``,
``action_class_dict["haar"] is Haar`` (and likewise for trim).

File: no new file created; demo.py deleted. Registry correctness proven by test suite.

- [x] **Step 2: Commit**

```bash
git rm src/qmp/algorithms/demo.py tests/unit/algorithms/test_demo.py
git commit -m "refactor: remove demo action; actions validated via per-algo registration tests"
```

---

### Task 5: Rewrite `__main__.py`

**Files:**
- Rewrite: `src/qmp/__main__.py`

- [ ] **Step 1: Write new `__main__.py`**

```python
"""qmp-kit CLI entry point.

Usage::

    qmp --config config.yaml --action.name haar --action.params.sampling_count=2048
"""

from __future__ import annotations

import argparse
import importlib
import logging
import typing
from dataclasses import dataclass, field

import dacite
import tyro
from omegaconf import OmegaConf

from qmp.algorithms._registry import action_class_dict, action_config_dict

logger = logging.getLogger(__name__)


@dataclass
class ActionCLI:
    name: str = "haar"
    params: dict[str, typing.Any] = field(default_factory=dict)


@dataclass
class ConfigCLI:
    action: ActionCLI = field(default_factory=lambda: ActionCLI(name="haar"))


def _load_yaml(path: str) -> ConfigCLI:
    raw = OmegaConf.load(path)
    plain = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(plain, dict)
    action = ActionCLI(**plain.get("action", {}))
    return ConfigCLI(action=action)


def main(argv: list[str] | None = None) -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    known, remaining = pre_parser.parse_known_args(argv)

    defaults = ConfigCLI()
    if known.config:
        defaults = _load_yaml(known.config)

    cli = tyro.cli(ConfigCLI, default=defaults, args=remaining)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("action=%s params=%s", cli.action.name, cli.action.params)

    importlib.import_module(f"qmp.algorithms.{cli.action.name}")
    cfg_cls = action_config_dict[cli.action.name]
    cfg = dacite.from_dict(cfg_cls, cli.action.params)
    impl_cls = action_class_dict[cli.action.name]
    instance = impl_cls(cfg)  # ty: ignore — dynamic dispatch
    instance.run()  # ty: ignore — dynamic dispatch


if __name__ == "__main__":
    main()
```

File: `src/qmp/__main__.py`

- [ ] **Step 2: Commit**

```bash
git add src/qmp/__main__.py
git commit -m "refactor: simplify CLI to action-only dispatch"
```

---

### Task 6: Verify end-to-end

- [ ] **Step 1: Create test YAML**

```yaml
action:
  name: haar
  params:
    message: "E2E test"
    model:
      name: hubbard
      params:
        m: 2
        n: 2
        t: 1.0
        u: 4.0
```

Save as `/tmp/test_config.yaml`

- [ ] **Step 2: Run CLI**

```bash
uv run qmp --config /tmp/test_config.yaml
```

Expected output: action=haar is dispatched (requires model/network to run; config-only smoke passes).

- [ ] **Step 3: Run CLI with override**

```bash
uv run qmp --config /tmp/test_config.yaml --action.params.message "override"
```

Expected: "Message: override"

- [ ] **Step 4: Run all tests**

```bash
uv run ruff format . && uv run ruff check src/qmp/ && uv run ty check src tests && uv run pytest tests/unit/ -q
```

Expected: all clean, all tests pass.

- [ ] **Step 5: Commit any final fixes**

---

### Task 7: Update AGENTS.md

**Files:**
- Update: `src/qmp/algorithms/_registry.py` (for the `build_from_ref` import path documentation — none needed, already in docstring)
- Create: `src/qmp/algorithms/AGENTS.md`

- [ ] **Step 1: Write algorithms AGENTS.md**

```markdown
# Algorithms 子系统

## Registry

- ``action_config_dict[name] = ConfigClass`` — 配置 dataclass 类型
- ``action_class_dict[name] = ImplClass`` — 实现类

每个算法模块在底部注册，上方通过 ``importlib.import_module`` 触发。

## Action config 约定

每个 action config 可包含 ``model: SubConfigRef | None`` 和 ``network: SubConfigRef | None`` 字段。
``None`` 表示该 action 不需要对应子系统。构造实例使用 ``build_from_ref`` helper。
```

File: `src/qmp/algorithms/AGENTS.md`

- [ ] **Step 2: Update root AGENTS.md to mention CLI architecture**

Add after the `## 代码结构` tree a brief note about CLI:

```markdown
## CLI

`qmp --config config.yaml` 入口在 `__main__.py`。YAML 仅含 `action` 顶层节，model/network 下沉为 action params 内的 `SubConfigRef`。

三个子系统各自维护双注册表 (`xxx_config_dict` + `xxx_class_dict`)，模块导入时自注册。
```

- [ ] **Step 3: Commit**

```bash
git add src/qmp/algorithms/AGENTS.md AGENTS.md
git commit -m "docs: add algorithms AGENTS.md; update root AGENTS with CLI arch"
```

---

## Self-Review

1. **Spec coverage**: All sections covered — YAML structure (Task 5), SubConfigRef (Task 1), registry (Tasks 2,3), build_from_ref (Task 1), action registry (Task 4), CLI dispatch (Task 5), AGENTS update (Task 7).
2. **Placeholder scan**: No TBD/TODO/placeholders. All code is explicit.
3. **Type consistency**: `SubConfigRef`, `build_from_ref`, `action_config_dict`, `action_class_dict` are consistent across all tasks.
