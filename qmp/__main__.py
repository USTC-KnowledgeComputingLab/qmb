"""
Main entry point for the qmp command-line interface.
"""

import logging
import pathlib
import importlib
import dacite
import hydra
import omegaconf
import torch

from .utility.context import RuntimeContext, DACITE_CAST
from .utility.action_dict import action_dict
from .utility.distributed import (
    DistributedConfig,
    spawn_workers,
    init_rpc_worker,
    shutdown_rpc,
    get_rank,
    get_world_size,
    get_local_device,
    parse_device_addr,
    is_rank_zero,
)


def run_main(runtime_config: omegaconf.DictConfig) -> None:
    """
    Execute the qmp application based on the provided configuration.
    This function is called by rank 0 (orchestrator) after RPC initialization.
    """
    # 0. Dynamic Imports
    importlib.import_module(f".algorithms.{runtime_config.action.name}", package=__package__)
    importlib.import_module(f".models.{runtime_config.model.name}", package=__package__)

    # 1. Setup Runtime Context
    context = dacite.from_dict(
        data_class=RuntimeContext,
        data=omegaconf.OmegaConf.to_container(runtime_config.common, resolve=True),  # type: ignore[arg-type]
        config=dacite.Config(cast=DACITE_CAST),
    )
    checkpoint_data = context.setup()

    # 2. Instantiate Algorithm
    run = dacite.from_dict(
        data_class=action_dict[runtime_config.action.name],  # type: ignore[arg-type]
        data=omegaconf.OmegaConf.to_container(runtime_config.action.params, resolve=True),  # type: ignore[arg-type]
        config=dacite.Config(cast=DACITE_CAST),
    )

    # 3. Execute Algorithm
    # The algorithm is responsible for creating its own models/networks using the context and config.
    run.main(context=context, runtime_config=runtime_config, checkpoint_data=checkpoint_data)


def worker_main(rank: int, world_size: int, runtime_config: omegaconf.DictConfig, master_addr: str, master_port: int) -> None:
    """
    Worker process entry point.

    Parameters
    ----------
    rank : int
        Global rank of this process.
    world_size : int
        Total number of processes.
    runtime_config : omegaconf.DictConfig
        Runtime configuration.
    master_addr : str
        Master node address.
    master_port : int
        Master port.
    """
    # Get device for this rank
    distributed_config = runtime_config.common.get("distributed", {})
    devices = distributed_config.get("devices", ["localhost:cuda:0"])
    _, device = parse_device_addr(devices[rank])

    # Initialize RPC
    init_rpc_worker(rank, world_size, device, master_addr, master_port)

    if rank == 0:
        # Rank 0 runs the main algorithm
        logging.info("Rank 0 (orchestrator) starting main algorithm")
        run_main(runtime_config)
    else:
        # Other ranks wait for RPC calls
        logging.info("Rank %d (worker) waiting for RPC calls", rank)

    # Shutdown RPC
    shutdown_rpc()


@hydra.main(version_base=None, config_path=str(pathlib.Path().resolve()), config_name="config")
def main(runtime_config: omegaconf.DictConfig) -> None:
    """
    Main entry point for qmp.

    Detects distributed configuration and spawns worker processes accordingly.
    Only rank 0 executes the main algorithm, other ranks serve as RPC workers.
    """
    config_dict = omegaconf.OmegaConf.to_container(runtime_config.common, resolve=True)

    # Get distributed configuration
    distributed = config_dict.get("distributed", {})
    devices = distributed.get("devices", ["localhost:cuda:0"])

    # Convert to DistributedConfig
    dist_config = DistributedConfig(
        devices=devices,
        master_port=distributed.get("master_port", 29500),
    )

    world_size = len(devices)
    master_addr = dist_config.master_addr
    master_port = dist_config.master_port

    logging.info("Distributed configuration: world_size=%d, master_addr=%s, master_port=%d", world_size, master_addr, master_port)

    # Spawn worker processes
    torch.multiprocessing.spawn(
        worker_main,
        args=(world_size, runtime_config, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()