"""
Distributed computing utilities using torch.distributed.rpc.

This module provides utilities for multi-node, multi-GPU distributed computing.
Each node spawns worker processes that bind to local GPUs. Only rank 0 (orchestrator)
runs the main algorithm loop, while other ranks serve as RPC workers.
"""

import os
import socket
import logging
import dataclasses
import typing
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp


@dataclasses.dataclass
class DistributedConfig:
    """
    Configuration for distributed computing.

    Devices are specified as "node_addr:device_type:index" strings, e.g.:
    - "10.0.0.1:cuda:0"
    - "localhost:cuda:1"

    Short form "device_type:index" is also supported and defaults to localhost:
    - "cuda:0" -> "localhost:cuda:0"
    - "cuda:2" -> "localhost:cuda:2"

    The first device in the list is the orchestrator (rank 0).
    """

    devices: list[str] = dataclasses.field(default_factory=lambda: ["cuda:0"])
    master_port: int = 29500

    @property
    def world_size(self) -> int:
        """Total number of processes across all nodes."""
        return len(self.devices)

    @property
    def master_addr(self) -> str:
        """Master node address (first device's node)."""
        return parse_device_addr(self.devices[0])[0]


def parse_device_addr(addr: str) -> tuple[str, torch.device]:
    """
    Parse a device address string into node address and torch device.

    Parameters
    ----------
    addr : str
        Device address in one of two formats:
        - Short form: "device_type:index", e.g., "cuda:0", "cuda:2"
          (defaults to localhost)
        - Full form: "node_addr:device_type:index", e.g., "10.0.0.1:cuda:0"

    Returns
    -------
    tuple[str, torch.device]
        (node_address, torch_device)
    """
    parts = addr.split(":")
    if len(parts) == 2:
        # Short form: "cuda:0" -> defaults to localhost
        node_addr = "localhost"
        device = torch.device(f"{parts[0]}:{parts[1]}")
    elif len(parts) == 3:
        # Full form: "10.0.0.1:cuda:0"
        node_addr = parts[0]
        device = torch.device(f"{parts[1]}:{parts[2]}")
    else:
        raise ValueError(f"Invalid device address format: {addr}. Expected 'device:index' or 'node:device:index'")
    return node_addr, device


def get_local_node_addr() -> str:
    """
    Get the local node's IP address.

    Returns
    -------
    str
        Local node IP address or hostname.
    """
    # Try to get IP address that can reach the master
    try:
        # Get hostname
        hostname = socket.gethostname()
        # Get IP address
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except Exception:
        return "localhost"


def get_local_devices(config: DistributedConfig) -> list[tuple[int, str, torch.device]]:
    """
    Get the devices that belong to the current local node.

    Parameters
    ----------
    config : DistributedConfig
        Distributed configuration.

    Returns
    -------
    list[tuple[int, str, torch.device]]
        List of (global_rank, node_addr, device) for local devices.
    """
    local_node = get_local_node_addr()
    local_devices = []

    for rank, addr in enumerate(config.devices):
        node_addr, device = parse_device_addr(addr)
        # Match by IP or hostname
        if node_addr == local_node or node_addr == "localhost" or node_addr == hostname_to_ip(local_node):
            local_devices.append((rank, node_addr, device))

    return local_devices


def hostname_to_ip(hostname: str) -> str:
    """Convert hostname to IP address if possible."""
    try:
        return socket.gethostbyname(hostname)
    except Exception:
        return hostname


def get_rank() -> int:
    """
    Get the current process's global rank.

    Returns
    -------
    int
        Global rank of current process.
    """
    # rpc uses internal mechanisms, we need to track rank ourselves
    # This is set during worker initialization
    return _CURRENT_RANK


def get_world_size() -> int:
    """
    Get the total number of processes in the distributed group.

    Returns
    -------
    int
        World size.
    """
    return _WORLD_SIZE


def get_local_device() -> torch.device:
    """
    Get the torch device for the current process.

    Returns
    -------
    torch.device
        Local device.
    """
    return _LOCAL_DEVICE


# Global state for current process
_CURRENT_RANK: int = 0
_WORLD_SIZE: int = 1
_LOCAL_DEVICE: torch.device = torch.device("cuda:0")


def _set_global_state(rank: int, world_size: int, device: torch.device) -> None:
    """Set the global distributed state for current process."""
    global _CURRENT_RANK, _WORLD_SIZE, _LOCAL_DEVICE
    _CURRENT_RANK = rank
    _WORLD_SIZE = world_size
    _LOCAL_DEVICE = device


def init_rpc_worker(rank: int, world_size: int, device: torch.device, master_addr: str, master_port: int) -> None:
    """
    Initialize RPC for a worker process.

    Parameters
    ----------
    rank : int
        Global rank of this process.
    world_size : int
        Total number of processes.
    device : torch.device
        Local device to use.
    master_addr : str
        Master node address.
    master_port : int
        Master port.
    """
    _set_global_state(rank, world_size, device)

    # Set CUDA device
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Initialize RPC
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        init_method=f"tcp://{master_addr}:{master_port}",
    )

    rpc.init_rpc(
        name=f"rank_{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options,
    )

    logging.info("RPC initialized: rank=%d, world_size=%d, device=%s", rank, world_size, device)


def shutdown_rpc() -> None:
    """Shutdown RPC for current process."""
    rpc.shutdown()


def spawn_workers(config: DistributedConfig, main_fn: callable, main_args: tuple = ()) -> None:
    """
    Spawn worker processes for the local node and run the main function on rank 0.

    Parameters
    ----------
    config : DistributedConfig
        Distributed configuration.
    main_fn : callable
        Function to run on rank 0 (orchestrator).
    main_args : tuple
        Arguments to pass to main_fn.
    """
    local_devices = get_local_devices(config)

    if len(local_devices) == 0:
        raise RuntimeError(f"No local devices found. Local node: {get_local_node_addr()}, Config devices: {config.devices}")

    logging.info("Local node: %s, local devices: %d", get_local_node_addr(), len(local_devices))

    world_size = config.world_size
    master_addr = config.master_addr
    master_port = config.master_port

    def worker_main(local_rank: int, local_devices: list, world_size: int, master_addr: str, master_port: int, main_fn: callable, main_args: tuple):
        """Worker process entry point."""
        global_rank, node_addr, device = local_devices[local_rank]

        init_rpc_worker(global_rank, world_size, device, master_addr, master_port)

        if global_rank == 0:
            # Rank 0 runs the main algorithm
            logging.info("Rank 0 starting main algorithm")
            main_fn(*main_args)
        else:
            # Other ranks just wait for RPC calls
            logging.info("Rank %d waiting for RPC calls", global_rank)

        shutdown_rpc()

    # Spawn processes
    mp.spawn(
        worker_main,
        args=(local_devices, world_size, master_addr, master_port, main_fn, main_args),
        nprocs=len(local_devices),
        join=True,
    )


def is_rank_zero() -> bool:
    """Check if current process is rank 0 (orchestrator)."""
    return get_rank() == 0


def rpc_remote_call(target_rank: int, fn: callable, args: tuple = ()) -> rpc.RRef:
    """
    Make a remote RPC call to a target rank.

    Parameters
    ----------
    target_rank : int
        Target rank to call.
    fn : callable
        Function to execute remotely.
    args : tuple
        Arguments to pass to the function.

    Returns
    -------
    rpc.RRef
        Remote reference to the result.
    """
    return rpc.remote(f"rank_{target_rank}", fn, args=args)


def rpc_sync_call(target_rank: int, fn: callable, args: tuple = ()) -> typing.Any:
    """
    Make a synchronous RPC call to a target rank.

    Parameters
    ----------
    target_rank : int
        Target rank to call.
    fn : callable
        Function to execute remotely.
    args : tuple
        Arguments to pass to the function.

    Returns
    -------
    Any
        Result of the remote call.
    """
    return rpc.rpc_sync(f"rank_{target_rank}", fn, args=args)