"""
This is the main entry point for the command line application.
"""

# ruff: noqa: F401

import pathlib
import hydra
import omegaconf
from .utility.subcommand_dict import subcommand_dict
from .utility.common import CommonConfig
from .utility.model_dict import model_dict
from .models import openfermion as _  # type: ignore[no-redef]
from .models import fcidump as _  # type: ignore[no-redef]
from .models import hubbard as _  # type: ignore[no-redef]
from .models import free_fermion as _  # type: ignore[no-redef]
from .models import ising as _  # type: ignore[no-redef]
from .algorithms import guide as _  # type: ignore[no-redef]
from .algorithms import vmc as _  # type: ignore[no-redef]
from .algorithms import haar as _  # type: ignore[no-redef]
from .algorithms import precompile as _  # type: ignore[no-redef]
from .algorithms import chop_imag as _  # type: ignore[no-redef]
from .algorithms import pert as _  # type: ignore[no-redef]
from .algorithms import pretrain as _  # type: ignore[no-redef]


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: omegaconf.DictConfig) -> None:
    """
    The main function for the command line application.
    """
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
    main()  # pylint: disable=no-value-for-parameter
