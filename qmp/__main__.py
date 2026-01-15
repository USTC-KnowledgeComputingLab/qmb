"""
Main entry point for the qmp command-line interface.
"""

import pathlib
import importlib
import torch
import dacite
import hydra
import omegaconf

from .utility.context import RuntimeContext
from .utility.subcommand_dict import subcommand_dict


@hydra.main(version_base=None, config_path=str(pathlib.Path().resolve()), config_name="config")
def main(runtime_config: omegaconf.DictConfig) -> None:
    """Execute the qmp application based on the provided configuration."""

    # 0. Dynamic Imports
    importlib.import_module(f".algorithms.{runtime_config.action.name}", package=__package__)
    importlib.import_module(f".models.{runtime_config.model.name}", package=__package__)

    # 1. Setup Runtime Context
    context = dacite.from_dict(
        data_class=RuntimeContext,
        data=omegaconf.OmegaConf.to_container(runtime_config.common, resolve=True),  # type: ignore[arg-type]
        config=dacite.Config(cast=[pathlib.Path, torch.device, tuple]),
    )
    checkpoint_data = context.setup()

    # 2. Instantiate Algorithm
    run = dacite.from_dict(
        data_class=subcommand_dict[runtime_config.action.name],  # type: ignore[arg-type]
        data=omegaconf.OmegaConf.to_container(runtime_config.action.params, resolve=True),  # type: ignore[arg-type]
        config=dacite.Config(cast=[pathlib.Path, torch.device, tuple]),
    )

    # 3. Execute Algorithm
    # The algorithm is responsible for creating its own models/networks using the context and config.
    run.main(context=context, runtime_config=runtime_config, checkpoint_data=checkpoint_data)


if __name__ == "__main__":
    main()
