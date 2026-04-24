"""
Main entry point for the qmp command-line interface.
"""

import logging
import pathlib
import importlib
import datetime
import os
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
    get_rank,
)


# Global debug log file path (set in main())
_DEBUG_LOG_FILE: pathlib.Path | None = None


def _debug_log(rank: int, msg: str) -> None:
    """Write a timestamped log message to debug file."""
    if _DEBUG_LOG_FILE is None:
        return
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    pid = os.getpid()
    with open(_DEBUG_LOG_FILE, "a") as f:
        f.write(f"[Rank {rank}] [PID {pid}] {ts} | {msg}\n")


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
    master_addr: str,
    master_port: int,
) -> None:
    """
    Worker process entry point for RPC workers.

    Parameters
    ----------
    local_rank : int
        Local rank within this node.
    local_devices : list[tuple[int, str, torch.device]]
        List of (global_rank, node_addr, device) for devices on this node.
    world_size : int
        Total number of processes across all nodes.
    master_addr : str
        Master node address.
    master_port : int
        Master port.
    """
    global_rank, node_addr, device = local_devices[local_rank]

    _debug_log(global_rank, "worker_main START")

    init_rpc_worker(global_rank, world_size, device, master_addr, master_port)
    _debug_log(global_rank, "RPC initialized, waiting for calls")

    shutdown_rpc()
    _debug_log(global_rank, "worker_main END")


@hydra.main(version_base=None, config_path=str(pathlib.Path().resolve()), config_name="config")
def main(runtime_config: omegaconf.DictConfig) -> None:
    """
    Main entry point for qmp.

    - If this node contains rank 0: main process runs as rank 0 (orchestrator),
      spawns (local_devices - 1) workers for RPC.
    - If this node does NOT contain rank 0: main process also runs as RPC worker,
      spawns (local_devices - 1) workers, total local_devices processes for RPC.
    """
    # Set debug log file path (in current working directory)
    global _DEBUG_LOG_FILE
    _DEBUG_LOG_FILE = pathlib.Path.cwd() / "qmp_debug.log"
    # Clear previous log
    if _DEBUG_LOG_FILE.exists():
        _DEBUG_LOG_FILE.unlink()

    config_dict = omegaconf.OmegaConf.to_container(runtime_config.common, resolve=True)

    devices = config_dict.get("devices", ["cuda:0"])
    master_port = config_dict.get("master_port", 29500)

    dist_config = DistributedConfig(
        devices=devices,
        master_port=master_port,
    )

    world_size = len(devices)
    master_addr = dist_config.master_addr

    local_devices = get_local_devices(dist_config)

    if len(local_devices) == 0:
        logging.warning("No local devices found for this node. Local node: %s, Config devices: %s",
                        get_local_node_addr(), devices)
        return

    # Check if rank 0 is on this node
    rank_0_on_this_node = any(rank == 0 for rank, _, _ in local_devices)

    logging.info("Distributed configuration: world_size=%d, master_addr=%s, master_port=%d",
                 world_size, master_addr, master_port)
    logging.info("Local node: %s, local devices: %d, rank_0_on_this_node: %s",
                 get_local_node_addr(), len(local_devices), rank_0_on_this_node)
    logging.info("Debug log file: %s", _DEBUG_LOG_FILE)

    _debug_log(-1, f"main() START, world_size={world_size}, local_devices={len(local_devices)}")

    # Spawn workers (excluding the rank that main process will handle)
    spawn_count = len(local_devices) - 1
    spawn_context = None

    if spawn_count > 0:
        # Devices for spawned workers: skip the first one (handled by main process)
        spawn_devices = local_devices[1:]
        spawn_context = torch.multiprocessing.spawn(
            worker_main,
            args=(spawn_devices, world_size, master_addr, master_port),
            nprocs=spawn_count,
            join=False,
        )
        logging.info("Spawned %d worker processes for ranks: %s",
                     spawn_count, [rank for rank, _, _ in spawn_devices])
        _debug_log(-1, f"spawned {spawn_count} workers for ranks {[rank for rank, _, _ in spawn_devices]}")

    # Main process handles the first local device
    main_rank, main_node, main_device = local_devices[0]

    _debug_log(main_rank, "main process START (will handle this rank)")

    init_rpc_worker(main_rank, world_size, main_device, master_addr, master_port)
    _debug_log(main_rank, "RPC initialized")

    if main_rank == 0:
        _debug_log(main_rank, "run_main START")
        run_main(runtime_config)
        _debug_log(main_rank, "run_main END")
    else:
        _debug_log(main_rank, "waiting for RPC calls")

    shutdown_rpc()
    _debug_log(main_rank, "main process END")

    # Wait for spawned workers to finish
    if spawn_context is not None:
        spawn_context.join()
    _debug_log(-1, "main() END, all workers finished")


if __name__ == "__main__":
    main()