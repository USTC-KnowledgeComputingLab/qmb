"""
This file contains the common step to create a model and network for various scripts.
"""

import sys
import logging
import typing
import pathlib
import dataclasses
import torch
from .model_dict import model_dict, ModelProto, NetworkProto
from .random_engine import dump_random_engine_state, load_random_engine_state


@dataclasses.dataclass
class CommonConfig:
    """
    This class defines the common settings needed to create a model and network.
    """

    # The model name
    model_name: str
    # The network name
    network_name: str
    # Arguments for physical model
    physics_args: tuple[str, ...] = ()
    # Arguments for network
    network_args: tuple[str, ...] = ()

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

    def main(
        self, *, model_param: typing.Any = None, network_param: typing.Any = None
    ) -> tuple[ModelProto, NetworkProto, typing.Any]:
        """
        The main function to create the model and network.
        """

        model_t = model_dict[self.model_name]
        model_config_t = model_t.config_t  # noqa: F841
        network_config_t = model_t.network_dict[self.network_name]  # noqa: F841

        self.folder().mkdir(parents=True, exist_ok=True)

        logging.info("Starting script with arguments: %a", sys.argv)
        logging.info("Model: %s, Network: %s", self.model_name, self.network_name)
        logging.info("Log directory: %s", self.folder())
        if model_param is not None:
            logging.info("Model parameters: %a", model_param)
        else:
            logging.info("Physics arguments: %a", self.physics_args)
        if network_param is not None:
            logging.info("Network parameters: %a", network_param)
        else:
            logging.info("Network arguments: %a", self.network_args)

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

        if model_param is None:
            raise ValueError(
                "model_param must be provided when calling main(). "
                "This should be an instance of the model's config class, "
                "typically created from Hydra configuration."
            )
        else:
            logging.info("The model parameters are given as %a, skipping parsing model arguments", model_param)
        logging.info("Loading the model")
        model: ModelProto = model_t(model_param)
        logging.info("Physical model loaded successfully")

        if network_param is None:
            raise ValueError(
                "network_param must be provided when calling main(). "
                "This should be an instance of the network's config class, "
                "typically created from Hydra configuration."
            )
        else:
            logging.info("The network parameters are given as %a, skipping parsing network arguments", network_param)
        logging.info("Initializing the network")
        network: NetworkProto = network_param.create(model)
        logging.info("Network initialized successfully")

        if "network" in data:
            logging.info("Loading state dict of the network")
            network.load_state_dict(data["network"])
        else:
            logging.info("Skipping loading state dict of the network")

        logging.info("Moving model to the device: %a", self.device)
        network = network.to(device=self.device, dtype=self.dtype)

        logging.info("Compiling the network")
        network = torch.jit.script(network)  # type: ignore[assignment]

        logging.info("The checkpoints will be saved every %d steps", self.checkpoint_interval)

        return model, network, data
