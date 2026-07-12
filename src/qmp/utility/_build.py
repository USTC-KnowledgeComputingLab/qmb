"""SubConfigRef and model construction helper.

``SubConfigRef`` is an (name, params) pair that action configs embed to
declare which model they need. ``build_model`` uses the global model
registry to construct instances from these refs.
"""

from __future__ import annotations

import importlib
import logging
import typing
from dataclasses import dataclass, field

import dacite

from qmp.models._model import model_config_dict, model_dict

logger = logging.getLogger(__name__)


@dataclass
class SubConfigRef:
    """Reference from an action config to a model."""

    name: str
    params: dict[str, typing.Any] = field(default_factory=dict)


def build_model(ref: SubConfigRef | None) -> typing.Any:
    """Construct a model instance from a SubConfigRef.

    Returns ``None`` if ``ref`` is ``None``. Otherwise lazy-imports the model
    module, deserialises ``ref.params`` into the registered config dataclass,
    and calls the implementation class.
    """
    if ref is None:
        return None
    if ref.name not in model_config_dict:
        try:
            importlib.import_module(f"qmp.models.{ref.name}")
        except ModuleNotFoundError:
            raise KeyError(f"Unknown model: {ref.name!r}") from None
    cfg_cls = model_config_dict[ref.name]
    cfg = dacite.from_dict(cfg_cls, ref.params)
    impl_cls = model_dict[ref.name]
    return impl_cls(cfg)
