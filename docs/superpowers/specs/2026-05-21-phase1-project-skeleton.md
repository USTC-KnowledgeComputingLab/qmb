# Phase 1: Project Skeleton Migration

**Date**: 2026-05-21
**Status**: approved

## Goal

Migrate qmp-kit from setuptools+setuptools_scm to hatchling+hatch-vcs build system. Create a clean project skeleton on an empty branch, from which features will be incrementally ported.

## Decisions

| Decision | Value |
|----------|-------|
| Build backend | `hatchling` |
| Version management | `hatch-vcs` (git tag driven) |
| Package layout | `src/qmp/` (src layout) |
| Python requirement | `>=3.12` |
| License | `AGPL-3.0-only` |
| Dependencies | None initially (add as features are ported) |
| Scripts/CLI | None initially (add when ported) |
| Pre-commit | Removed (not in skeleton) |
| C++ format | Google style (`BasedOnStyle: Google`) |

## Files to Create

```
qmp/
├── pyproject.toml
├── .gitignore
├── .clang-format
├── LICENSE.md                    # AGPL v3 from GNU website
└── src/
    └── qmp/
        └── __init__.py
```

## pyproject.toml Structure

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "qmp-kit"
dynamic = ["version"]
requires-python = ">=3.12"
authors = [{ name = "Hao Zhang", email = "hzhangxyz@outlook.com" }]
description = "Quantum Manybody Problem"
license = "AGPL-3.0-only"
keywords = ["quantum", "manybody", "quantum-chemistry"]
classifiers = [
  "Programming Language :: Python :: 3",
  "Topic :: Scientific/Engineering :: Physics",
]
dependencies = []

[project.urls]
Homepage = "https://github.com/USTC-KnowledgeComputingLab/qmp-kit"
Repository = "https://github.com/USTC-KnowledgeComputingLab/qmp-kit.git"

[tool.hatch.version]
source = "vcs"

[tool.hatch.version.raw-options]
version_scheme = "no-guess-dev"
local_scheme = "no-local-version"

[tool.hatch.build.hooks.vcs]
version-file = "src/qmp/_version.py"
```

## .gitignore

Minimal Python gitignore covering cache, build artifacts, auto-generated version file, and IDE config.

## .clang-format

Google style base, ready for future C++/CUDA kernel migration.

## Out of Scope

- Pre-commit configuration
- README, CONTRIBUTING
- Any source code beyond `__init__.py`
- tests/ directory