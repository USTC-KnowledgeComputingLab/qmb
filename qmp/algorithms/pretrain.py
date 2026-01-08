"""
This file implements a pretraining for quantum many-body problems.
"""

import typing
import logging
import dataclasses
import torch
from ..utility import losses
from ..utility.common import CommonConfig
from ..utility.optimizer import initialize_optimizer
from ..utility.subcommand_dict import subcommand_dict


@dataclasses.dataclass
class PretrainConfig:
    """
    Configuration for pretraining quantum many-body models.
    """

    common: CommonConfig

    # Dataset path for pretraining
    dataset_path: str
    # The learning rate for the local optimizer
    learning_rate: float = 1e-3
    # The name of the loss function to use
    loss_name: str = "sum_filtered_angle_scaled_log"

    def main(self, *, model_param: typing.Any = None, network_param: typing.Any = None) -> None:
        """
        The main function for pretraining.
        """

        model, network, data = self.common.main(model_param=model_param, network_param=network_param)

        dataset = torch.load(self.dataset_path, map_location="cpu", weights_only=True)
        config = dataset[0].to(device=self.common.device)
        psi = dataset[1].to(device=self.common.device)

        optimizer = initialize_optimizer(
            network.parameters(),
            use_lbfgs=False,
            learning_rate=self.learning_rate,
            state_dict=data.get("optimizer"),
        )

        if "pretrain" not in data:
            data["pretrain"] = {"global": 0}

        loss_func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(losses, self.loss_name)

        while True:

            def closure():
                optimizer.zero_grad()
                prediction = network(config)
                loss = loss_func(psi, prediction)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            prediction = network(config)  # noqa: F841
            logging.info("Step %d: Loss = %.6f", data["pretrain"]["global"], loss.item())

            logging.info("Saving model checkpoint")
            data["pretrain"]["global"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            self.common.save(data, data["pretrain"]["global"])
            logging.info("Checkpoint successfully saved")

            logging.info("Current optimization cycle completed")


subcommand_dict["pretrain"] = PretrainConfig
