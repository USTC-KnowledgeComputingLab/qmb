"""Main entry point for the qmp command-line interface."""

# ruff: noqa: F401

import pathlib

import hydra
import omegaconf

from .algorithms import (
    chop_imag,
    guide,
    haar,
    pert,
    precompile,
    pretrain,
    vmc,
)
from .models import fcidump, free_fermion, hubbard, ising, openfermion
from .utility.common import CommonConfig
from .utility.model_dict import model_dict
from .utility.subcommand_dict import subcommand_dict


@hydra.main(version_base=None, config_path=str(pathlib.Path().resolve()), config_name="config")
def main(config: omegaconf.DictConfig) -> None:
    """Execute the qmp application based on the provided configuration."""
    action = subcommand_dict[config.action.name]
    common = CommonConfig(
        log_path=pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir),
        model_name=config.model.name,
        network_name=config.network.name,
        **config.common,
    )
    run = action(
        common=common,
        **config.action.params,
    )

    model_t = model_dict[config.model.name]
    model_config_t = model_t.config_t
    model_param = model_config_t(**config.model.params)
    network_config_t = model_t.network_dict[config.network.name]
    network_param = network_config_t(**config.network.params)

    if config.action.name == "guide":
        run.main(model_param=model_param, network_param=network_param, config=config)  # type: ignore[call-arg]
    else:
        run.main(model_param=model_param, network_param=network_param)  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
