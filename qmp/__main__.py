"""Main entry point for the qmp command-line interface."""

import pathlib

import hydra
import omegaconf
from hydra.utils import instantiate

from .algorithms import chop_imag, guide, haar, pert, pretrain, vmc  # noqa: F401
from .models import fcidump, hubbard, ising, openfermion  # noqa: F401
from .utility.common import RuntimeContext
from .utility.subcommand_dict import subcommand_dict


@hydra.main(version_base=None, config_path=str(pathlib.Path().resolve()), config_name="config")
def main(config: omegaconf.DictConfig) -> None:
    """Execute the qmp application based on the provided configuration."""

    # 1. Setup Runtime Context
    ctx = instantiate(
        config.common,
        _target_=RuntimeContext,
        log_path=pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir),
    )
    checkpoint_data = ctx.setup()

    # 2. Instantiate Algorithm
    run = instantiate(
        config.action.params,
        _target_=subcommand_dict[config.action.name],
    )

    # 3. Execute Algorithm
    # The algorithm is responsible for creating its own models/networks using the context and config.
    run.main(ctx=ctx, config=config, checkpoint_data=checkpoint_data)


if __name__ == "__main__":
    main()
