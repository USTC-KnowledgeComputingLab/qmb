"""Network registry.

``network_config_dict`` maps network names to their config dataclass type.
``network_class_dict`` maps network names to their implementation class.
Networks register themselves at import time.
"""

from __future__ import annotations

network_config_dict: dict[str, type[object]] = {}
network_class_dict: dict[str, type[object]] = {}
