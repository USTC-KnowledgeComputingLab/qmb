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
    name: str = "demo"
    params: dict[str, typing.Any] = field(default_factory=dict)


@dataclass
class ConfigCLI:
    action: ActionCLI = field(default_factory=lambda: ActionCLI(name="demo"))


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
    if cli.action.name not in action_config_dict:
        raise KeyError(f"Unknown action: {cli.action.name!r}. Registered: {list(action_config_dict)}")
    cfg_cls = action_config_dict[cli.action.name]
    cfg = dacite.from_dict(cfg_cls, cli.action.params)
    impl_cls = action_class_dict[cli.action.name]
    instance = impl_cls(cfg)
    instance.run()


if __name__ == "__main__":
    main()
