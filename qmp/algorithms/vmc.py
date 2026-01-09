"""
This file implements a variational Monte Carlo method for solving quantum many-body problems.
"""

import logging
import typing
import dataclasses
import omegaconf
import torch
import torch.utils.tensorboard
from ..utility.context import RuntimeContext
from ..utility.subcommand_dict import subcommand_dict


@dataclasses.dataclass
class VmcConfig:
    """
    The VMC optimization for solving quantum many-body problems.
    """

    # The sampling count
    sampling_count: int = 4000
    # The number of relative configurations to be used in energy calculation
    relative_count: int = 40000
    # The number of steps for the local optimizer
    local_step: int = 1000

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        """
        The main function for the VMC optimization.
        """

        # Create Model and Network
        model = context.create_model(runtime_config.model)
        network = context.create_network(runtime_config.network, model, checkpoint_data.get("network"))
        data = checkpoint_data

        # Create Optimizer
        optimizer = context.create_optimizer(
            runtime_config.optimizer, network.parameters(), checkpoint_data.get("optimizer")
        )

        logging.info(
            "Arguments Summary: Sampling Count: %d, Relative Count: %d, Local Steps: %d, ",
            self.sampling_count,
            self.relative_count,
            self.local_step,
        )

        if "vmc" not in data:
            data["vmc"] = {"global": 0, "local": 0}

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=context.folder())  # type: ignore[no-untyped-call]

        while True:
            logging.info("Starting a new optimization cycle")

            logging.info("Sampling configurations")
            configs_i, psi_i, _, _ = network.generate_unique(self.sampling_count)
            logging.info("Sampling completed, unique configurations count: %d", len(configs_i))

            logging.info("Calculating relative configurations")
            if self.relative_count <= len(configs_i):
                configs_src = configs_i
                configs_dst = configs_i
            else:
                configs_src = configs_i
                configs_dst = torch.cat(
                    [configs_i, model.find_relative(configs_i, psi_i, self.relative_count - len(configs_i))]
                )
            logging.info("Relative configurations calculated, count: %d", len(configs_dst))

            def closure() -> torch.Tensor:
                # Optimizing energy
                optimizer.zero_grad()
                psi_src = network(configs_src)
                with torch.no_grad():
                    psi_dst = network(configs_dst)
                    hamiltonian_psi_dst = model.apply_within(configs_dst, psi_dst, configs_src)
                num = psi_src.conj() @ hamiltonian_psi_dst
                den = psi_src.conj() @ psi_src.detach()
                energy = num / den
                energy = energy.real
                energy.backward()  # type: ignore[no-untyped-call]
                return energy

            logging.info("Starting local optimization process")

            for i in range(self.local_step):
                energy: torch.Tensor = optimizer.step(closure)  # type: ignore[assignment,arg-type]
                logging.info(
                    "Local optimization in progress, step: %d, energy: %.10f, ref energy: %.10f",
                    i,
                    energy.item(),
                    model.ref_energy,
                )
                writer.add_scalar("vmc/energy", energy, data["vmc"]["local"])  # type: ignore[no-untyped-call]
                writer.add_scalar("vmc/error", energy - model.ref_energy, data["vmc"]["local"])  # type: ignore[no-untyped-call]
                data["vmc"]["local"] += 1

            logging.info("Local optimization process completed")

            writer.flush()  # type: ignore[no-untyped-call]

            logging.info("Saving model checkpoint")
            data["vmc"]["global"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            context.save(data, data["vmc"]["global"])
            logging.info("Checkpoint successfully saved")

            logging.info("Current optimization cycle completed")


subcommand_dict["vmc"] = VmcConfig
