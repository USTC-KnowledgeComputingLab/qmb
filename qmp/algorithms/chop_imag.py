"""
This file implements the subspace chopping for the result of the imag script.
"""

import logging
import typing
import dataclasses
import omegaconf
import torch.utils.tensorboard
from ..utility.context import RuntimeContext
from ..utility.subcommand_dict import subcommand_dict


@dataclasses.dataclass
class ChopImagConfig:
    """
    The subspace chopping for the result of the imag script.
    """

    # The number of configurations to eliminate every iteration
    chop_size: int = 10000
    # The estimated magnitude of the second order term
    second_order_magnitude: float = 0.0

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        """
        The main function for the subspace chopping.
        """

        model = context.create_model(runtime_config.model)
        data = checkpoint_data

        logging.info(
            "Arguments Summary: Chop Size: %d, Second Order Magnitude: %.10f",
            self.chop_size,
            self.second_order_magnitude,
        )

        configs, psi = data["imag"]["pool"]
        configs = configs.to(device=context.device)
        psi = psi.to(device=context.device)

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=context.folder())  # type: ignore[no-untyped-call]

        original_configs = configs
        original_psi = psi
        ordered_configs: list[torch.Tensor] = []
        ordered_psi: list[torch.Tensor] = []
        mapping: dict[int, tuple[float, float]] = {}

        i = 0
        while True:
            num_configs = len(configs)
            logging.info("The number of configurations: %d", num_configs)
            writer.add_scalar("chop_imag/num_configs", num_configs, i)  # type: ignore[no-untyped-call]
            psi = psi / psi.norm()
            hamiltonian_psi = model.apply_within(configs, psi, configs)
            psi_hamiltonian_psi = (psi.conj() @ hamiltonian_psi).real
            energy = psi_hamiltonian_psi
            logging.info(
                "The energy: %.10f, The energy error is %.10f", energy.item(), energy.item() - model.ref_energy
            )
            writer.add_scalar("chop_imag/energy", energy.item(), i)  # type: ignore[no-untyped-call]
            writer.add_scalar("chop_imag/error", energy.item() - model.ref_energy, i)  # type: ignore[no-untyped-call]
            writer.flush()  # type: ignore[no-untyped-call]
            mapping[num_configs] = (energy.item(), energy.item() - model.ref_energy)
            if self.second_order_magnitude >= 0:
                grad = hamiltonian_psi - psi_hamiltonian_psi * psi
                delta = -psi.conj() * grad
                real_delta = 2 * delta.real
                second_order = (psi.conj() * psi).real * self.second_order_magnitude
                rate = (real_delta + second_order).argsort()
            else:
                second_order = (psi.conj() * psi).real
                rate = second_order.argsort()
            unselected = rate[: self.chop_size]
            ordered_configs.append(configs[unselected])
            ordered_psi.append(psi[unselected])
            selected = rate[self.chop_size :]
            if len(selected) == 0:
                break
            configs = configs[selected]
            psi = psi[selected]
            i += 1

        data["chop_imag"] = {
            "ordered_configs": torch.cat(ordered_configs, dim=0),
            "ordered_psi": torch.cat(ordered_psi, dim=0),
            "original_configs": original_configs,
            "original_psi": original_psi,
            "mapping": mapping,
        }
        context.save(data, 0)


subcommand_dict["chop_imag"] = ChopImagConfig
