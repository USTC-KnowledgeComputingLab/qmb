"""Main entry point for the qmp command-line interface."""

import pathlib
import hydra
import omegaconf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from .algorithms import chop_imag, guide, haar, pert, pretrain, vmc  # noqa: F401
from .models import fcidump, hubbard, ising, openfermion  # noqa: F401
from .utility.context import RuntimeContext
from .utility.subcommand_dict import subcommand_dict


@hydra.main(version_base=None, config_path=str(pathlib.Path().resolve()), config_name="config")
def main(runtime_config: omegaconf.DictConfig) -> None:
    """Execute the qmp application based on the provided configuration."""

    # 1. Setup Runtime Context
    context = instantiate(
        runtime_config.common,
        _target_=RuntimeContext,
        log_path=pathlib.Path(HydraConfig.get().runtime.output_dir),
    )
    checkpoint_data = context.setup()

    # 2. Instantiate Algorithm
    run = instantiate(
        runtime_config.action.params,
        _target_=subcommand_dict[runtime_config.action.name],
    )

    # 3. Execute Algorithm
    # The algorithm is responsible for creating its own models/networks using the context and config.
    run.main(context=context, runtime_config=runtime_config, checkpoint_data=checkpoint_data)


if __name__ == "__main__":
    main()
