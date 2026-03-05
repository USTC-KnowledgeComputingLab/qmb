"""
This module is used to store a dictionary that maps action names to their corresponding dataclass types.

Other packages or subpackages can register their actions by adding entries to this dictionary, such as
```
from qmp.utility.action_dict import action_dict
action_dict["my_action"] = MyAction
```
"""

import typing
import omegaconf
from .context import RuntimeContext


class ActionProto(typing.Protocol):
    """
    This protocol defines a dataclass with a `main` method, which will be called when the action is executed.
    """

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        """
        The main method to be called when the action is executed.
        """


action_dict: dict[str, type[ActionProto]] = {}
