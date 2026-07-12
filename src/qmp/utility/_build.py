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
    config_dict: dict[str, typing.Any],
    impl_dict: dict[str, typing.Any],
) -> typing.Any:
    """Construct a subsystem instance from a SubConfigRef.

    If ``ref`` is ``None``, returns ``None`` (action doesn't need this subsystem).
    Otherwise, lazy-imports the module, deserialises ``ref.params`` into the
    registered config dataclass, and calls the implementation class with it.
    """
    if ref is None:
        return None
    if ref.name not in config_dict:
        try:
            importlib.import_module(f"qmp.{subsystem}.{ref.name}")
        except ModuleNotFoundError:
            raise KeyError(f"Unknown {subsystem}: {ref.name!r}") from None
    cfg_cls = config_dict[ref.name]
    cfg = dacite.from_dict(cfg_cls, ref.params)
    impl_cls = impl_dict[ref.name]
    return impl_cls(cfg)
