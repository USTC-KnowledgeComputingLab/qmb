"""qmp-kit CLI entry point.

Usage::

    qmp --config config.yaml --model.name hubbard --action.name demo
    qmp --model.name fcidump --model.params.model_path=/tmp/test.fcidump --action.name demo
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

from qmp.algorithms._registry import action_dict
from qmp.models._model import model_config_dict, model_dict

logger = logging.getLogger(__name__)


# ==============================================================================
# CLI config schema
# ==============================================================================


@dataclass
class ModelCLI:
    """Model selection. ``name`` maps to ``model_dict`` keys; ``params`` are passed to the model constructor."""

    name: str = "fcidump"
    params: dict[str, typing.Any] = field(default_factory=dict)


@dataclass
class ActionCLI:
    """Algorithm selection.

    ``name`` maps to ``action_dict`` keys; ``params`` are passed
    to the algorithm constructor via dacite.
    """

    name: str = "demo"
    params: dict[str, typing.Any] = field(default_factory=dict)


@dataclass
class CommonCLI:
    """Global runtime settings."""

    seed: int = 2333
    device: str = "auto"


@dataclass
class ConfigCLI:
    """Top-level configuration for ``qmp`` CLI."""

    model: ModelCLI = field(default_factory=ModelCLI)
    action: ActionCLI = field(default_factory=ActionCLI)
    common: CommonCLI = field(default_factory=CommonCLI)


# ==============================================================================
# YAML loading with ${} interpolation
# ==============================================================================


def _load_yaml(path: str) -> ConfigCLI:
    """Load a YAML config file with OmegaConf (supports ``${}`` interpolation)."""
    raw = OmegaConf.load(path)
    plain = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(plain, dict)
    # dacite-style: hand-roll nested construction for the 3 known sections
    model = ModelCLI(**plain.get("model", {}))
    action = ActionCLI(**plain.get("action", {}))
    common = CommonCLI(**plain.get("common", {}))
    return ConfigCLI(model=model, action=action, common=common)


# ==============================================================================
# Dispatch
# ==============================================================================


def _import_model(name: str) -> None:
    """Dynamically import the model module to trigger registration."""
    importlib.import_module(f".{name}", package="qmp.models")


def _import_action(name: str) -> None:
    """Dynamically import the algorithm module to trigger registration."""
    importlib.import_module(f".{name}", package="qmp.algorithms")


def _build_model(cli: ModelCLI) -> typing.Any:
    _import_model(cli.name)
    if cli.name not in model_dict:
        raise KeyError(f"Unknown model: {cli.name!r}. Registered: {list(model_dict)}")

    config_cls = model_config_dict.get(cli.name)
    if config_cls is None:
        raise KeyError(f"Model {cli.name!r} has no registered config class.")
    config = dacite.from_dict(config_cls, cli.params)
    model_cls = model_dict[cli.name]
    return model_cls(config)


def _build_action(cli: ActionCLI) -> typing.Any:
    _import_action(cli.name)
    if cli.name not in action_dict:
        raise KeyError(f"Unknown action: {cli.name!r}. Registered: {list(action_dict)}")
    config_cls = action_dict[cli.name]

    config = dacite.from_dict(config_cls, cli.params)
    mod = importlib.import_module(f"qmp.algorithms.{cli.name}")
    instance = mod.Demo(config)
    return instance


# ==============================================================================
# Main
# ==============================================================================


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (exposed via ``pyproject.toml [project.scripts]``)."""
    # 1. Extract --config before tyro processes the rest
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    known, remaining = pre_parser.parse_known_args(argv)

    # 2. Load YAML defaults (with ${} interpolation)
    defaults = ConfigCLI()
    if known.config:
        defaults = _load_yaml(known.config)

    # 3. tyro parses remaining CLI args, overriding YAML values
    cli = tyro.cli(ConfigCLI, default=defaults, args=remaining)

    # 4. Build and run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("model=%s params=%s", cli.model.name, cli.model.params)
    logger.info("action=%s params=%s", cli.action.name, cli.action.params)

    model = _build_model(cli.model)
    action = _build_action(cli.action)
    action.run(model)  # ty: ignore — dynamic dispatch from registry dict


if __name__ == "__main__":
    main()
