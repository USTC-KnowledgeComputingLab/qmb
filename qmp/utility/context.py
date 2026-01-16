"""
This file contains the common step to create a model and network for various scripts.
"""

import sys
import logging
import typing
import pathlib
import dataclasses
import torch
import dacite
import omegaconf
from hydra.core.hydra_config import HydraConfig
from .model_dict import model_dict, ModelProto, NetworkProto
from .random_engine import dump_random_engine_state, load_random_engine_state
from .optimizer import migrate_optimizer


@dataclasses.dataclass
class RuntimeContext:
    """
    This class defines the common runtime environment.
    """

    # The parent path to load the checkpoint from, leave empty to use the current folder or start from scratch
    parent_path: pathlib.Path | None = None
    # The manual random seed, leave empty for set seed automatically
    random_seed: int | None = None
    # The interval to save the checkpoint
    checkpoint_interval: int = 5
    # The device to run on
    device: torch.device = torch.device(type="cuda", index=0)
    # The dtype of the network, leave empty to skip modifying the dtype
    dtype: str | torch.dtype | None = None
    # The maximum absolute step for the process, leave empty to loop forever
    max_absolute_step: int | None = None
    # The maximum relative step for the process, leave empty to loop forever
    max_relative_step: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.dtype, str):
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
        return pathlib.Path(HydraConfig.get().runtime.output_dir)

    def setup(self) -> dict[str, typing.Any]:
        """
        Setup the runtime environment, and returns the loaded checkpoint data (if any).
        """
        logging.info("Log directory: %s", self.folder())
        self.folder().mkdir(parents=True, exist_ok=True)

        logging.info("Disabling PyTorch's default gradient computation")
        torch.set_grad_enabled(False)

        logging.info("Attempting to load checkpoint")
        data: typing.Any = {}
        checkpoint_path = self.folder() / "data.pth" if self.parent_path is None else self.parent_path
        if checkpoint_path.exists():
            logging.info("Checkpoint found at: %s", checkpoint_path)
            data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            logging.info("Checkpoint loaded successfully")
        else:
            if self.parent_path is not None:
                raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            logging.info("Checkpoint not found at: %s, start from scratch", checkpoint_path)

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
            logging.info("Reached the maximum step, exiting")
            sys.exit(0)

    def create_model(self, model_config: omegaconf.DictConfig) -> ModelProto:
        """
        Create a model instance from the configuration.
        """
        model_t = model_dict[model_config.name]
        logging.info("Loading the model: %s", model_config.name)
        # Instantiate the parameters first
        model_param = dacite.from_dict(
            data_class=model_t.config_t,
            data=omegaconf.OmegaConf.to_container(model_config.params, resolve=True),  # type: ignore[arg-type]
            config=dacite.Config(cast=[pathlib.Path, torch.device, tuple]),
        )
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
        network_config_t = model.network_dict[network_name]

        logging.info("Initializing the network: %s", network_name)
        network_param = dacite.from_dict(
            data_class=network_config_t,
            data=omegaconf.OmegaConf.to_container(network_config.params, resolve=True),  # type: ignore[arg-type]
            config=dacite.Config(cast=[pathlib.Path, torch.device, tuple]),
        )
        network: NetworkProto = network_param.create(model)
        logging.info("Network initialized successfully")

        if state_dict is not None:
            logging.info("Loading state dict of the network")
            network.load_state_dict(state_dict)
        else:
            logging.info("Skipping loading state dict of the network")

        logging.info("Moving network to device: %s", self.device)
        assert not isinstance(self.dtype, str)
        network = network.to(device=self.device, dtype=self.dtype)

        logging.info("Compiling the network")
        network = torch.jit.script(network)  # type: ignore[assignment]

        return network

    def create_optimizer(
        self,
        optimizer_config: omegaconf.DictConfig,
        params: typing.Iterable[torch.Tensor],
        state_dict: dict[str, typing.Any] | None = None,
    ) -> torch.optim.Optimizer:
        """
        Create an optimizer instance from the configuration.

        Args:
            optimizer_config: The optimizer configuration.
            params: The parameters to optimize.
            state_dict: Optional state dict to load into the optimizer.
        """
        logging.info("Initializing the optimizer")

        optimizer_t = getattr(torch.optim, optimizer_config.name)
        optimizer = optimizer_t(params=params, **optimizer_config.params)  # type: ignore[arg-type]

        if state_dict is not None:
            logging.info("Loading state dict of the optimizer")
            optimizer.load_state_dict(state_dict)
            migrate_optimizer(optimizer)
        else:
            logging.info("Skipping loading state dict of the optimizer")

        return optimizer
