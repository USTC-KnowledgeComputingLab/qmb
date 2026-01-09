"""
This file implements a pretraining for quantum many-body problems.
"""

import typing
import logging
import dataclasses
import omegaconf
import torch
from ..utility import losses
from ..utility.context import RuntimeContext
from ..utility.subcommand_dict import subcommand_dict


@dataclasses.dataclass
class PretrainConfig:
    """
    Configuration for pretraining quantum many-body models.
    """

    # Dataset path for pretraining
    dataset_path: str
    # The name of the loss function to use
    loss_name: str = "sum_filtered_angle_scaled_log"

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        """
        The main function for pretraining.
        """

        model = context.create_model(runtime_config.model)
        network = context.create_network(runtime_config.network, model, checkpoint_data.get("network"))
        data = checkpoint_data

        dataset = torch.load(self.dataset_path, map_location="cpu", weights_only=True)
        config_tensor = dataset[0].to(device=context.device)
        psi = dataset[1].to(device=context.device)

        # Create Optimizer
        optimizer = context.create_optimizer(
            runtime_config.optimizer, network.parameters(), checkpoint_data.get("optimizer")
        )

        if "pretrain" not in data:
            data["pretrain"] = {"global": 0}

        loss_func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(losses, self.loss_name)

        while True:

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                prediction = network(config_tensor)
                loss = loss_func(psi, prediction)
                loss.backward()  # type: ignore[no-untyped-call]
                return loss

            loss: torch.Tensor = optimizer.step(closure)  # type: ignore[assignment, arg-type]
            # prediction = network(config_tensor) # Unused?
            logging.info("Step %d: Loss = %.6f", data["pretrain"]["global"], loss.item())

            logging.info("Saving model checkpoint")
            data["pretrain"]["global"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            context.save(data, data["pretrain"]["global"])
            logging.info("Checkpoint successfully saved")

            logging.info("Current optimization cycle completed")


subcommand_dict["pretrain"] = PretrainConfig
