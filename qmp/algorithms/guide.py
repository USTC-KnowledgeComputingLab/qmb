"""
This file implements a variational Monte Carlo method for solving quantum many-body problems with guide.
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
class GuideConfig:
    """
    The guided VMC optimization for solving quantum many-body problems.
    """

    # The sampling count
    sampling_count: int = 4000
    # The number of relative configurations to be used in energy calculation
    relative_count: int = 40000
    # The number of steps for the local optimizer
    local_step: int = 1000
    # The number of steps for the distribution optimizer
    dist_step: int = 100

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        """
        The main function for the guided VMC optimization.
        """

        # Create Model
        model = context.create_model(runtime_config.model)

        # Create Main Network
        network = context.create_network(runtime_config.network, model, checkpoint_data.get("network"))

        # Create Sampling Network
        sampling = context.create_network(runtime_config.sampling, model, checkpoint_data.get("sampling"))

        data = checkpoint_data

        # Create Optimizers
        # We use the same optimizer configuration for both network and sampling
        optimizer_network = context.create_optimizer(
            runtime_config.optimizer, network.parameters(), checkpoint_data.get("optimizer")
        )
        optimizer_sampling = context.create_optimizer(
            runtime_config.optimizer, sampling.parameters(), checkpoint_data.get("optimizer_sampling")
        )

        logging.info(
            "Arguments Summary: Sampling Count: %d, Relative Count: %d, Local Steps: %d, Dist Steps: %d",
            self.sampling_count,
            self.relative_count,
            self.local_step,
            self.dist_step,
        )

        if "guide" not in data:
            data["guide"] = {"global": 0, "local": 0, "dist": 0}

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=context.folder())  # type: ignore[no-untyped-call]

        while True:
            logging.info("Starting a new optimization cycle")

            logging.info("Sampling configurations")
            configs_src_network, psi_src_network, _, _ = network.generate_unique(self.sampling_count)
            configs_src_sampling, psi_src_sampling, _, _ = sampling.generate_unique(self.sampling_count)

            logging.info("Calculating relative configurations")
            if self.relative_count <= len(configs_src_network):
                configs_dst_network = configs_src_network
            else:
                configs_dst_network = torch.cat(
                    [
                        configs_src_network,
                        model.find_relative(
                            configs_src_network, psi_src_network, self.relative_count - len(configs_src_network)
                        ),
                    ]
                )
            if self.relative_count <= len(configs_src_sampling):
                configs_dst_sampling = configs_src_sampling
            else:
                configs_dst_sampling = torch.cat(
                    [
                        configs_src_sampling,
                        model.find_relative(
                            configs_src_sampling, psi_src_sampling, self.relative_count - len(configs_src_sampling)
                        ),
                    ]
                )

            def energy_sampling() -> torch.Tensor:
                configs_src = configs_src_sampling
                configs_dst = configs_dst_sampling
                psi_src = network(configs_src)
                with torch.no_grad():
                    psi_dst = network(configs_dst)
                    hamiltonian_psi_dst = model.apply_within(configs_dst, psi_dst, configs_src)
                num = psi_src.conj() @ hamiltonian_psi_dst
                den = psi_src.conj() @ psi_src.detach()
                energy = num / den
                energy.real.backward()  # type: ignore[no-untyped-call]
                return energy.real

            def energy_network() -> torch.Tensor:
                with torch.no_grad():
                    configs_src = configs_src_network
                    configs_dst = configs_dst_network
                    psi_src = network(configs_src)
                    psi_dst = network(configs_dst)
                    hamiltonian_psi_dst = model.apply_within(configs_dst, psi_dst, configs_src)
                    num = psi_src.conj() @ hamiltonian_psi_dst
                    den = psi_src.conj() @ psi_src.detach()
                    energy = num / den
                    return energy.real

            def distribution() -> torch.Tensor:
                configs_src = configs_src_sampling
                configs_dst = configs_dst_sampling
                with torch.no_grad():
                    psi_src = network(configs_src)
                    psi_dst = network(configs_dst)
                    hamiltonian_psi_dst = model.apply_within(configs_dst, psi_dst, configs_src)
                    num = psi_src.conj() @ hamiltonian_psi_dst
                    den = psi_src.conj() @ psi_src
                    energy = num / den
                    local_energy = hamiltonian_psi_dst / psi_src
                    energy_diff = local_energy - energy
                    weight = (energy_diff.real**2 + energy_diff.imag**2) * (psi_src.real**2 + psi_src.imag**2)
                    weight = weight / weight.sum()
                pred_amplitude = sampling(configs_src)
                pred_weight = pred_amplitude.real**2 + pred_amplitude.imag**2
                pred_weight = pred_weight / pred_weight.sum()
                loss = pred_weight - weight
                total_loss = (loss**2).sum()
                total_loss.backward()  # type: ignore[no-untyped-call]
                return total_loss

            logging.info("Starting local optimization process")

            for i in range(self.local_step):
                optimizer_network.zero_grad()
                sampling_energy: torch.Tensor = optimizer_network.step(energy_sampling)  # type: ignore[assignment,arg-type]
                network_energy: torch.Tensor = energy_network()
                logging.info(
                    "Local optimization in progress, step: %d, energy from sampling: %.10f, energy from network: %.10f, ref energy: %.10f",
                    i,
                    sampling_energy.item(),
                    network_energy.item(),
                    model.ref_energy,
                )
                writer.add_scalar("guide/energy/sampling", sampling_energy, data["guide"]["local"])  # type: ignore[no-untyped-call]
                writer.add_scalar("guide/error/sampling", sampling_energy - model.ref_energy, data["guide"]["local"])  # type: ignore[no-untyped-call]
                writer.add_scalar("guide/energy/network", network_energy, data["guide"]["local"])  # type: ignore[no-untyped-call]
                writer.add_scalar("guide/error/network", network_energy - model.ref_energy, data["guide"]["local"])  # type: ignore[no-untyped-call]
                for _ in range(self.dist_step):
                    optimizer_sampling.zero_grad()
                    dist_loss: torch.Tensor = optimizer_sampling.step(distribution)  # type: ignore[assignment,arg-type]
                    writer.add_scalar("guide/dist/loss", dist_loss, data["guide"]["dist"])  # type: ignore[no-untyped-call]
                    data["guide"]["dist"] += 1
                data["guide"]["local"] += 1

            logging.info("Local optimization process completed")

            writer.flush()  # type: ignore[no-untyped-call]

            logging.info("Saving model checkpoint")
            data["guide"]["global"] += 1
            data["network"] = network.state_dict()
            data["sampling"] = sampling.state_dict()
            data["optimizer"] = optimizer_network.state_dict()
            data["optimizer_sampling"] = optimizer_sampling.state_dict()
            context.save(data, data["guide"]["global"])
            logging.info("Checkpoint successfully saved")

            logging.info("Current optimization cycle completed")


subcommand_dict["guide"] = GuideConfig
