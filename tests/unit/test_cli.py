"""Tests for CLI entry point and dispatch."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

from qmp.__main__ import ActionCLI, ConfigCLI, _load_yaml, main

# ---- ConfigCLI defaults ----


def test_config_cli_defaults() -> None:
    cfg = ConfigCLI()
    assert cfg.action.name == "demo"
    assert cfg.action.params == {}


def test_action_cli() -> None:
    cli = ActionCLI(name="haar", params={"k": 32})
    assert cli.name == "haar"
    assert cli.params == {"k": 32}


# ---- YAML loading ----


def test_load_yaml_basic() -> None:
    yaml_content = """action:
  name: haar
  params:
    lr: 0.01
"""
    tmp = Path(tempfile.gettempdir()) / "test_cli_basic.yaml"
    tmp.write_text(yaml_content)
    cfg = _load_yaml(str(tmp))
    assert cfg.action.name == "haar"
    assert cfg.action.params == {"lr": 0.01}


def test_load_yaml_default_action() -> None:
    """Empty YAML → use ActionCLI defaults."""
    yaml_content = ""
    tmp = Path(tempfile.gettempdir()) / "test_cli_empty.yaml"
    tmp.write_text(yaml_content)
    cfg = _load_yaml(str(tmp))
    assert cfg.action.name == "demo"
    assert cfg.action.params == {}


def test_load_yaml_missing_action_section() -> None:
    """YAML without action key → defaults."""
    yaml_content = "foo: bar\n"
    tmp = Path(tempfile.gettempdir()) / "test_cli_no_action.yaml"
    tmp.write_text(yaml_content)
    cfg = _load_yaml(str(tmp))
    assert cfg.action.name == "demo"


def test_load_yaml_some_params() -> None:
    """Partial params → only specified keys are present."""
    yaml_content = """action:
  name: demo
  params:
    message: "hello"
    nothing: else
"""
    tmp = Path(tempfile.gettempdir()) / "test_cli_partial.yaml"
    tmp.write_text(yaml_content)
    cfg = _load_yaml(str(tmp))
    assert cfg.action.params == {"message": "hello", "nothing": "else"}


# ---- main() dispatch ----


def test_main_no_config() -> None:
    """main() with no args invokes demo action."""
    with mock.patch("qmp.algorithms.demo.Demo.run") as mock_run:
        main(argv=[])
        mock_run.assert_called_once()


def test_main_with_yaml() -> None:
    """main() with --config loads YAML and invokes action."""
    yaml_content = """action:
  name: demo
  params:
    message: "from_yaml"
"""
    tmp = Path(tempfile.gettempdir()) / "test_cli_main.yaml"
    tmp.write_text(yaml_content)
    with mock.patch("qmp.algorithms.demo.Demo.run") as mock_run:
        main(argv=["--config", str(tmp)])
        mock_run.assert_called_once()


def test_main_override() -> None:
    """CLI --action.params... overrides YAML."""
    yaml_content = """action:
  name: demo
  params:
    message: "from_yaml"
"""
    tmp = Path(tempfile.gettempdir()) / "test_cli_override.yaml"
    tmp.write_text(yaml_content)
    with mock.patch("qmp.algorithms.demo.Demo.run") as mock_run:
        main(argv=["--config", str(tmp), "--action.params.message", "overridden"])
        mock_run.assert_called_once()
