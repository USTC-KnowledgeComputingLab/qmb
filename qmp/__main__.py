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
    get_local_devices,
    get_local_node_addr,
    init_rpc_worker,
    shutdown_rpc,
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


def worker_main(
    local_rank: int,
    local_devices: list[tuple[int, str, torch.device]],
    world_size: int,
    runtime_config: omegaconf.DictConfig,
    master_addr: str,
    master_port: int,
) -> None:
    """
    Worker process entry point.

    Parameters
    ----------
    local_rank : int
        Local rank within this node (0 to len(local_devices)-1).
    local_devices : list[tuple[int, str, torch.device]]
        List of (global_rank, node_addr, device) for devices on this node.
    world_size : int
        Total number of processes across all nodes.
    runtime_config : omegaconf.DictConfig
        Runtime configuration.
    master_addr : str
        Master node address.
    master_port : int
        Master port.
    """
    # Get global rank and device from local_devices
    global_rank, node_addr, device = local_devices[local_rank]

    # Initialize RPC
    init_rpc_worker(global_rank, world_size, device, master_addr, master_port)

    if global_rank == 0:
        # Rank 0 runs the main algorithm
        logging.info("Rank 0 (orchestrator) starting main algorithm")
        run_main(runtime_config)
    else:
        # Other ranks wait for RPC calls
        logging.info("Rank %d (worker) waiting for RPC calls", global_rank)

    # Shutdown RPC
    shutdown_rpc()


@hydra.main(version_base=None, config_path=str(pathlib.Path().resolve()), config_name="config")
def main(runtime_config: omegaconf.DictConfig) -> None:
    """
    Main entry point for qmp.

    Each node spawns only the worker processes for its local devices.
    Only rank 0 executes the main algorithm, other ranks serve as RPC workers.
    """
    config_dict = omegaconf.OmegaConf.to_container(runtime_config.common, resolve=True)

    # Get distributed configuration
    distributed = config_dict.get("distributed", {})
    devices = distributed.get("devices", ["cuda:0"])

    # Convert to DistributedConfig
    dist_config = DistributedConfig(
        devices=devices,
        master_port=distributed.get("master_port", 29500),
    )

    world_size = len(devices)
    master_addr = dist_config.master_addr
    master_port = dist_config.master_port

    # Get devices that belong to this node
    local_devices = get_local_devices(dist_config)

    if len(local_devices) == 0:
        logging.warning("No local devices found for this node. Local node: %s, Config devices: %s",
                        get_local_node_addr(), devices)
        return

    logging.info("Distributed configuration: world_size=%d, master_addr=%s, master_port=%d",
                 world_size, master_addr, master_port)
    logging.info("Local node: %s, spawning %d worker(s) for ranks: %s",
                 get_local_node_addr(), len(local_devices),
                 [rank for rank, _, _ in local_devices])

    # Spawn only local worker processes
    torch.multiprocessing.spawn(
        worker_main,
        args=(local_devices, world_size, runtime_config, master_addr, master_port),
        nprocs=len(local_devices),
        join=True,
    )


if __name__ == "__main__":
    main()