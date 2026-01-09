"""
This file contains the common step to create a model and network for various scripts.
"""

import sys
import logging
import typing
import pathlib
import dataclasses
import omegaconf
import torch
from hydra.utils import instantiate
from .model_dict import model_dict, ModelProto, NetworkProto
from .random_engine import dump_random_engine_state, load_random_engine_state


@dataclasses.dataclass
class RuntimeContext:
    """
    This class defines the common runtime environment (logging, device, random seed, checkpoints).
    """

    # The log path
    log_path: pathlib.Path = pathlib.Path("logs")
    # The log path for parent job job name, it is only used for loading the checkpoint from the parent job, leave empty to use the current job name
    parent_path: pathlib.Path | None = None
    # The manual random seed, leave empty for set seed automatically
    random_seed: int | None = None
    # The interval to save the checkpoint
    checkpoint_interval: int = 5
    # The device to run on
    device: torch.device = torch.device(type="cuda", index=0)
    # The dtype of the network, leave empty to skip modifying the dtype
    dtype: torch.dtype | None = None
    # The maximum absolute step for the process, leave empty to loop forever
    max_absolute_step: int | None = None
    # The maximum relative step for the process, leave empty to loop forever
    max_relative_step: int | None = None

    def __post_init__(self) -> None:
        if self.log_path is not None:
            self.log_path = pathlib.Path(self.log_path)
        if self.parent_path is not None:
            self.parent_path = pathlib.Path(self.parent_path)
        if self.device is not None:
            self.device = torch.device(self.device)
        if self.dtype is not None:
            match self.dtype:
                case "bfloat16":
                    self.dtype = torch.bfloat16
                case "float16" | "half":
                    self.dtype = torch.float16
                case "float32" | "float":
                    self.dtype = torch.float32
                case "float64" | "double":
                    self.dtype = torch.float64
                case _:
                    raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.max_absolute_step is not None and self.max_relative_step is not None:
            raise ValueError("Both max_absolute_step and max_relative_step are set, please set only one of them.")

    def folder(self) -> pathlib.Path:
        """
        Get the folder for storing logs.
        """
        return self.log_path

    def setup(self) -> dict[str, typing.Any]:
        """
        Setup the runtime environment (directories, logging, seed, initial checkpoint load).
        Returns the loaded checkpoint data (if any).
        """
        self.folder().mkdir(parents=True, exist_ok=True)

        logging.info("Starting script with arguments: %a", sys.argv)
        logging.info("Log directory: %s", self.folder())
        logging.info("Disabling PyTorch's default gradient computation")
        torch.set_grad_enabled(False)

        logging.info("Attempting to load checkpoint")
        data: typing.Any = {}
        checkpoint_path = (self.folder() if self.parent_path is None else self.parent_path) / "data.pth"
        if checkpoint_path.exists():
            logging.info("Checkpoint found at: %s, loading...", checkpoint_path)
            data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            logging.info("Checkpoint loaded successfully")
        else:
            if self.parent_path is not None:
                raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            logging.info("Checkpoint not found at: %s", checkpoint_path)

        if self.random_seed is not None:
            logging.info("Setting random seed to: %d", self.random_seed)
            torch.manual_seed(self.random_seed)
        elif "random" in data:
            logging.info("Loading random seed from the checkpoint")
            torch.set_rng_state(data["random"]["host"])
            if data["random"]["device_type"] == self.device.type:
                load_random_engine_state(data["random"]["device"], self.device)
            else:
                logging.info("Skipping loading random engine state for device since the device type does not match")
        else:
            logging.info("Random seed not specified, using current seed: %d", torch.seed())

        logging.info("The checkpoints will be saved every %d steps", self.checkpoint_interval)
        return data

    def save(self, data: typing.Any, step: int) -> None:
        """
        Save data to checkpoint.
        """
        data["random"] = {
            "host": torch.get_rng_state(),
            "device": dump_random_engine_state(self.device),
            "device_type": self.device.type,
        }
        data_path = self.folder() / "data.pth"
        local_data_path = self.folder() / f"data.{step}.pth"
        torch.save(data, local_data_path)
        data_path.unlink(missing_ok=True)
        if step % self.checkpoint_interval == 0:
            data_path.symlink_to(f"data.{step}.pth")
        else:
            local_data_path.rename(data_path)
        if self.max_relative_step is not None:
            self.max_absolute_step = step + self.max_relative_step - 1
            self.max_relative_step = None
        if step == self.max_absolute_step:
            logging.info("Reached the maximum step, exiting.")
            sys.exit(0)

    def create_model(self, model_config: omegaconf.DictConfig) -> ModelProto:
        """
        Create a model instance from the configuration.
        """
        model_t = model_dict[model_config.name]
        logging.info("Loading the model: %s", model_config.name)
        # Instantiate the parameters first
        model_param = instantiate(model_config.params, _target_=model_t.config_t)
        # Then create the model
        model: ModelProto = model_t(model_param)
        logging.info("Physical model loaded successfully")
        return model

    def create_network(
        self,
        network_config: omegaconf.DictConfig,
        model: ModelProto,
        state_dict: dict[str, typing.Any] | None = None,
    ) -> NetworkProto:
        """
        Create a network instance from the configuration.

        Args:
            network_config: The network configuration part (e.g., config.network).
            model: The physics model instance.
            state_dict: Optional state dict to load into the network.
        """
        network_name = network_config.name
        model_cls = type(model)
        if not hasattr(model_cls, "network_dict"):
            raise ValueError(f"Model class {model_cls} does not have 'network_dict'.")

        network_config_t = getattr(model_cls, "network_dict")[network_name]

        logging.info("Initializing the network: %s", network_name)
        network_param = instantiate(network_config.params, _target_=network_config_t)
        network: NetworkProto = network_param.create(model)

        if state_dict is not None:
            logging.info("Loading state dict of the network")
            network.load_state_dict(state_dict)
        else:
            logging.info("Skipping loading state dict of the network")

        logging.info("Moving network to device: %s", self.device)
        network = network.to(device=self.device, dtype=self.dtype)

        logging.info("Compiling the network")
        network = torch.jit.script(network)  # type: ignore[assignment]

        logging.info("Network initialized successfully")
        return network
