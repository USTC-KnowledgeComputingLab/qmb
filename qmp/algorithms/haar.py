"This file implements a two-step optimization process for solving quantum many-body problems based on imaginary time."

import copy
import logging
import typing
import dataclasses
import functools
import omegaconf
import scipy
import torch
import torch.utils.tensorboard
from ..utility import losses
from ..utility.common import RuntimeContext
from ..utility.subcommand_dict import subcommand_dict
from ..utility.model_dict import ModelProto
from ..utility.optimizer import initialize_optimizer, scale_learning_rate


@dataclasses.dataclass
class _DynamicLanczos:
    """
    This class implements the dynamic Lanczos algorithm for solving quantum many-body problems.
    """

    model: ModelProto
    configs: torch.Tensor
    psi: torch.Tensor
    step: int
    threshold: float
    count_extend: int
    batch_count_apply_within: int
    single_extend: bool
    first_extend: bool

    def _extend(self, psi: torch.Tensor, basic_configs: torch.Tensor | None = None) -> None:
        if basic_configs is None:
            basic_configs = self.configs
        logging.info("Extending basis...")

        count_core = len(self.configs)
        logging.info("Number of core configurations: %d", count_core)

        self.configs = torch.cat(
            [self.configs, self.model.find_relative(basic_configs, psi, self.count_extend, self.configs)]
        )
        count_selected = len(self.configs)
        self.psi = torch.nn.functional.pad(self.psi, (0, count_selected - count_core))
        logging.info("Basis extended from %d to %d", count_core, count_selected)

    def run(self) -> typing.Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Run the Lanczos algorithm.

        Yields
        ------
        energy : torch.Tensor
            The ground energy.
        configs : torch.Tensor
            The configurations.
        psi : torch.Tensor
            The wavefunction amplitude on the configurations.
        """
        alpha: list[torch.Tensor]
        beta: list[torch.Tensor]
        v: list[torch.Tensor]
        energy: torch.Tensor
        psi: torch.Tensor
        if self.count_extend == 0:
            # Do not extend the configuration, process the standard lanczos.
            for _, [alpha, beta, v] in zip(range(1 + self.step), self._run()):
                energy, psi = self._eigh_tridiagonal(alpha, beta, v)
                yield energy, self.configs, psi
        elif self.first_extend:
            # Extend the configuration before all iterations.
            psi = self.psi
            for _ in range(self.step):
                selected = (psi.conj() * psi).real.argsort(descending=True)[: self.count_extend]
                configs = self.configs
                self._extend(psi[selected], self.configs[selected])
                psi = self.model.apply_within(configs, psi, self.configs)
            for _, [alpha, beta, v] in zip(range(1 + self.step), self._run()):
                energy, psi = self._eigh_tridiagonal(alpha, beta, v)
                yield energy, self.configs, psi
        elif self.single_extend:
            # Extend the configuration only once after the whole iteration.
            for _, [alpha, beta, v] in zip(range(1 + self.step), self._run()):
                energy, psi = self._eigh_tridiagonal(alpha, beta, v)
                yield energy, self.configs, psi
            # Extend based on all vector in v.
            v_sum = (
                functools.reduce(torch.add, ((vi.conj() * vi).real.cpu() for vi in v))
                .sqrt()
                .to(device=self.configs.device)
            )
            self._extend(v_sum)
            for _, [alpha, beta, v] in zip(range(1 + self.step), self._run()):
                energy, psi = self._eigh_tridiagonal(alpha, beta, v)
                yield energy, self.configs, psi
        else:
            # Extend the configuration, during processing the dynamic lanczos.
            for step in range(1 + self.step):
                for _, [alpha, beta, v] in zip(range(1 + step), self._run()):
                    pass
                energy, psi = self._eigh_tridiagonal(alpha, beta, v)
                yield energy, self.configs, psi
                if step != self.step:
                    self._extend(v[-1])

    def _run(self) -> typing.Iterable[tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]]:
        """
        Process the standard lanczos.

        Yields
        ------
        alpha : list[torch.Tensor]
            The alpha values.
        beta : list[torch.Tensor]
            The beta values.
        v : list[torch.Tensor]
            The v values.
        """
        # In this function, we distribute data to the GPU and CPU.
        # The details are as follows:
        # All data other than v is always on the GPU.
        # The last v is always on the GPU and the rest are moved to the CPU immediately after necessary calculations.
        v: list[torch.Tensor] = [self.psi / torch.linalg.norm(self.psi)]
        alpha: list[torch.Tensor] = []
        beta: list[torch.Tensor] = []
        w: torch.Tensor
        w = self._apply_within(self.configs, v[-1], self.configs)
        alpha.append((w.conj() @ v[-1]).real)
        yield (alpha, beta, v)
        w = w - alpha[-1] * v[-1]
        while True:
            norm_w = torch.linalg.norm(w)
            if norm_w < self.threshold:
                break
            beta.append(norm_w)
            v.append(w / beta[-1])
            w = self._apply_within(self.configs, v[-1], self.configs)
            alpha.append((w.conj() @ v[-1]).real)
            yield (alpha, beta, v)
            w = w - alpha[-1] * v[-1] - beta[-1] * v[-2]
            v[-2] = v[-2].cpu()  # v maybe very large, so we need to move it to CPU

    def _apply_within(self, configs_i: torch.Tensor, psi_i: torch.Tensor, configs_j: torch.Tensor) -> torch.Tensor:
        batch_size = len(configs_i)
        local_batch_size = batch_size // self.batch_count_apply_within
        remainder = batch_size % self.batch_count_apply_within
        result: torch.Tensor | None = None
        for i in range(self.batch_count_apply_within):
            if i < remainder:
                current_local_batch_size = local_batch_size + 1
            else:
                current_local_batch_size = local_batch_size
            start_index = i * local_batch_size + min(i, remainder)
            end_index = start_index + current_local_batch_size
            local_result = self.model.apply_within(
                configs_i[start_index:end_index],
                psi_i[start_index:end_index],
                configs_j,
            )
            if result is None:
                result = local_result
            else:
                result = result + local_result
        assert result is not None
        return result

    def _eigh_tridiagonal(
        self,
        alpha: list[torch.Tensor],
        beta: list[torch.Tensor],
        v: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(beta) == 0:
            return alpha[0], v[0]
        # Currently, PyTorch does not support eigh_tridiagonal natively, so we resort to using SciPy for this operation.
        # We can only use 'stebz' or 'stemr' drivers in the current version of SciPy.
        # However, 'stemr' consumes a lot of memory, so we opt for 'stebz' here.
        # 'stebz' is efficient and only takes a few seconds even for large matrices with dimensions up to 10,000,000.
        vals, vecs = scipy.linalg.eigh_tridiagonal(
            torch.stack(alpha, dim=0).cpu(),
            torch.stack(beta, dim=0).cpu(),
            lapack_driver="stebz",
            select="i",
            select_range=(0, 0),
        )
        energy = torch.as_tensor(vals[0])
        result = functools.reduce(
            torch.add, (weight[0] * vector.to(device=self.configs.device) for weight, vector in zip(vecs, v))
        )
        return energy, result


def _sampling_from_last_iteration(
    pool: tuple[torch.Tensor, torch.Tensor] | None, number: int
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """
    Sample configurations and wavefunction amplitudes from the last iteration.

    Parameters
    ----------
    pool : tuple[torch.Tensor, torch.Tensor] | None
        The pool of configurations and wavefunction amplitudes from the last iteration.
    number : int
        The number of samples to draw.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] | tuple[None, None]
        A tuple containing the sampled configurations and wavefunction amplitudes, or (None, None) if the pool is empty.
    """
    if pool is None:
        return None, None
    configs, psi = pool
    probabilities = (psi.conj() * psi).real
    phi = probabilities.log()
    g = phi - (-torch.rand_like(phi).log()).log()
    _, indices = torch.topk(g, min(number, len(g)))
    return configs[indices], psi[indices]


def _merge_pool_from_neural_network_and_pool_from_last_iteration(
    configs_a: torch.Tensor,
    psi_a: torch.Tensor,
    configs_b: torch.Tensor | None,
    psi_b: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge the pool of configurations and psi values from the neural network and the last iteration.

    Parameters
    ----------
    configs_a : torch.Tensor
        Configurations from the neural network.
    psi_a : torch.Tensor
        Psi values from the neural network.
    configs_b : torch.Tensor | None
        Configurations from the last iteration.
    psi_b : torch.Tensor | None
        Psi values from the last iteration.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Merged configurations and psi values.
    """
    if configs_b is None and psi_b is None:
        return configs_a, psi_a
    assert configs_b is not None
    assert psi_b is not None
    configs_both = torch.cat([configs_a, configs_b])
    psi_both = torch.cat([psi_a, psi_b])
    configs_result, indices = torch.unique(configs_both, return_inverse=True, dim=0)
    # If the configurations are not unique, we prefer the psi from the last iteration.
    psi_result = torch.zeros(len(configs_result), device=psi_both.device, dtype=psi_both.dtype).scatter_(
        0, indices, psi_both
    )
    return configs_result, psi_result


@dataclasses.dataclass
class HaarConfig:
    """
    The two-step optimization process for solving quantum many-body problems based on imaginary time.
    """

    # The sampling count from neural network
    sampling_count_from_neural_network: int = 1024
    # The sampling count from last iteration
    sampling_count_from_last_iteration: int = 1024
    # The extend count for the Krylov subspace
    krylov_extend_count: int = -1
    # Whether to extend Krylov subspace before all iterations
    krylov_extend_first: bool = False
    # Whether to extend only once for Krylov subspace
    krylov_single_extend: bool = False
    # The number of Krylov iterations to perform
    krylov_iteration: int = 32
    # The threshold for the Krylov iteration
    krylov_threshold: float = 1e-8
    # The name of the loss function to use
    loss_name: str = "sum_filtered_angle_scaled_log"
    # Whether to use the global optimizer
    global_opt: bool = False
    # Whether to use LBFGS instead of Adam
    use_lbfgs: bool = False
    # The learning rate for the local optimizer
    learning_rate: float = -1
    # The number of steps for the local optimizer
    local_step: int = -1
    # The early break loss threshold for local optimization
    local_loss: float = 1e-8
    # The number of psi values to log after local optimization
    logging_psi: int = 30
    # The local batch count used to avoid memory overflow in generating configurations
    local_batch_count_generation: int = 1
    # The local batch count used to avoid memory overflow in apply within
    local_batch_count_apply_within: int = 1
    # The local batch count used to avoid memory overflow in loss function
    local_batch_count_loss_function: int = 1

    def __post_init__(self) -> None:
        if self.learning_rate == -1:
            self.learning_rate = 1 if self.use_lbfgs else 1e-3
        if self.local_step == -1:
            self.local_step = 1000 if self.use_lbfgs else 10000
        if self.krylov_extend_count == -1:
            self.krylov_extend_count = 2048 if self.krylov_single_extend else 64

    def main(
        self,
        ctx: RuntimeContext,
        config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        """
        The main function of two-step optimization process based on imaginary time.
        """

        model = ctx.create_model(config.model)
        network = ctx.create_network(config.network, model, checkpoint_data.get("network"))
        data = checkpoint_data

        logging.info(
            "Arguments Summary: "
            "Sampling Count From neural network: %d, "
            "Sampling Count From Last iteration: %d, "
            "Krylov Extend Count: %d, "
            "Krylov Extend First: %s, "
            "Krylov Single Extend: %s, "
            "Krylov Iteration: %d, "
            "Krylov Threshold: %.10f, "
            "Loss Function: %s, "
            "Global Optimizer: %s, "
            "Use LBFGS: %s, "
            "Learning Rate: %.10f, "
            "Local Steps: %d, "
            "Local Loss Threshold: %.10f, "
            "Logging Psi: %d, "
            "Local Batch Count For Generation: %d, "
            "Local Batch Count For Apply Within: %d, "
            "Local Batch Count For Loss Function: %d",
            self.sampling_count_from_neural_network,
            self.sampling_count_from_last_iteration,
            self.krylov_extend_count,
            "Yes" if self.krylov_extend_first else "No",
            "Yes" if self.krylov_single_extend else "No",
            self.krylov_iteration,
            self.krylov_threshold,
            self.loss_name,
            "Yes" if self.global_opt else "No",
            "Yes" if self.use_lbfgs else "No",
            self.learning_rate,
            self.local_step,
            self.local_loss,
            self.logging_psi,
            self.local_batch_count_generation,
            self.local_batch_count_apply_within,
            self.local_batch_count_loss_function,
        )

        optimizer = initialize_optimizer(
            network.parameters(),
            use_lbfgs=self.use_lbfgs,
            learning_rate=self.learning_rate,
            state_dict=data.get("optimizer"),
        )

        if "haar" not in data and "imag" in data:
            logging.warning("The 'imag' subcommand is deprecated, please use 'haar' instead.")
            data["haar"] = data["imag"]
            del data["imag"]
        if "haar" not in data:
            data["haar"] = {"global": 0, "local": 0, "lanczos": 0, "pool": None}
        else:
            pool_configs, pool_psi = data["haar"]["pool"]
            data["haar"]["pool"] = (pool_configs.to(device=ctx.device), pool_psi.to(device=ctx.device))

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=ctx.folder())  # type: ignore[no-untyped-call]

        while True:
            logging.info("Starting a new optimization cycle")

            logging.info("Sampling configurations from neural network")
            configs_from_neural_network, psi_from_neural_network, _, _ = network.generate_unique(
                self.sampling_count_from_neural_network, self.local_batch_count_generation
            )
            logging.info("Sampling configurations from last iteration")
            configs_from_last_iteration, psi_from_last_iteration = _sampling_from_last_iteration(
                data["haar"]["pool"], self.sampling_count_from_last_iteration
            )
            logging.info("Merging configurations from neural network and last iteration")
            configs, original_psi = _merge_pool_from_neural_network_and_pool_from_last_iteration(
                configs_from_neural_network,
                psi_from_neural_network,
                configs_from_last_iteration,
                psi_from_last_iteration,
            )
            logging.info("Sampling completed, unique configurations count: %d", len(configs))

            logging.info("Computing the target for local optimization")
            target_energy: torch.Tensor
            for target_energy, configs, original_psi in _DynamicLanczos(
                model=model,
                configs=configs,
                psi=original_psi,
                step=self.krylov_iteration,
                threshold=self.krylov_threshold,
                count_extend=self.krylov_extend_count,
                batch_count_apply_within=self.local_batch_count_apply_within,
                single_extend=self.krylov_single_extend,
                first_extend=self.krylov_extend_first,
            ).run():
                logging.info(
                    "The current energy is %.10f where the sampling count is %d", target_energy.item(), len(configs)
                )
                writer.add_scalar("haar/lanczos/energy", target_energy, data["haar"]["lanczos"])  # type: ignore[no-untyped-call]
                writer.add_scalar("haar/lanczos/error", target_energy - model.ref_energy, data["haar"]["lanczos"])  # type: ignore[no-untyped-call]
                data["haar"]["lanczos"] += 1
            max_index = original_psi.abs().argmax()
            target_psi = original_psi / original_psi[max_index]
            logging.info(
                "Local optimization target calculated, the target energy is %.10f, the sampling count is %d",
                target_energy.item(),
                len(configs),
            )

            loss_func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(losses, self.loss_name)

            optimizer = initialize_optimizer(
                network.parameters(),
                use_lbfgs=self.use_lbfgs,
                learning_rate=self.learning_rate,
                new_opt=not self.global_opt,
                optimizer=optimizer,
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                total_size = len(configs)
                batch_size = total_size // self.local_batch_count_loss_function
                remainder = total_size % self.local_batch_count_loss_function
                total_loss = 0.0
                total_psi = []
                for i in range(self.local_batch_count_loss_function):
                    if i < remainder:
                        current_batch_size = batch_size + 1
                    else:
                        current_batch_size = batch_size
                    start_index = i * batch_size + min(i, remainder)
                    end_index = start_index + current_batch_size
                    batch_indices = torch.arange(start_index, end_index, device=configs.device, dtype=torch.int64)
                    psi_batch = target_psi[batch_indices]
                    batch_indices = torch.cat(
                        (batch_indices, torch.tensor([max_index], device=configs.device, dtype=torch.int64))
                    )
                    batch_configs = configs[batch_indices]
                    psi = network(batch_configs)
                    psi_max = psi[-1]
                    psi = psi[:-1]
                    psi = psi / psi_max
                    loss = loss_func(psi, psi_batch)
                    loss = loss * (current_batch_size / total_size)
                    loss.backward()  # type: ignore[no-untyped-call]
                    total_loss += loss.item()
                    total_psi.append(psi.detach())
                total_loss_tensor = torch.tensor(total_loss)
                total_loss_tensor.psi = torch.cat(total_psi)  # type: ignore[attr-defined]
                return total_loss_tensor

            loss: torch.Tensor
            try_index = 0
            while True:
                state_backup = copy.deepcopy(network.state_dict())
                optimizer_backup = copy.deepcopy(optimizer.state_dict())

                logging.info("Starting local optimization process")
                success = True
                last_loss: float = 0.0
                local_step: int = data["haar"]["local"]
                scale_learning_rate(optimizer, 1 / (1 << try_index))
                for i in range(self.local_step):
                    loss = optimizer.step(closure)  # type: ignore[assignment,arg-type]
                    logging.info("Local optimization in progress, step %d, current loss: %.10f", i, loss.item())
                    writer.add_scalar(f"haar/loss/{self.loss_name}", loss, local_step)  # type: ignore[no-untyped-call]
                    local_step += 1
                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.warning("Loss is NaN, restoring the previous state and exiting the optimization loop")
                        success = False
                        break
                    if loss < self.local_loss:
                        logging.info("Local optimization halted as the loss threshold has been met")
                        break
                    if abs(loss - last_loss) < self.local_loss:
                        logging.info("Local optimization halted as the loss difference is too small")
                        break
                    last_loss = loss.item()
                scale_learning_rate(optimizer, 1 << try_index)
                if success:
                    if any(torch.isnan(param).any() or torch.isinf(param).any() for param in network.parameters()):
                        logging.warning(
                            "NaN detected in parameters, restoring the previous state and exiting the optimization loop"
                        )
                        success = False
                if success:
                    logging.info("Local optimization process completed")
                    data["haar"]["local"] = local_step
                    break
                network.load_state_dict(state_backup)
                optimizer.load_state_dict(optimizer_backup)
                try_index = try_index + 1

            logging.info("Current optimization cycle completed")

            loss = typing.cast(torch.Tensor, torch.enable_grad(closure)())  # type: ignore[no-untyped-call,call-arg]
            psi: torch.Tensor = loss.psi  # type: ignore[attr-defined]
            final_energy = ((psi.conj() @ model.apply_within(configs, psi, configs)) / (psi.conj() @ psi)).real
            logging.info(
                "Loss during local optimization: %.10f, Final energy: %.10f, Target energy: %.10f, Reference energy: %.10f, Final error: %.10f",
                loss.item(),
                final_energy.item(),
                target_energy.item(),
                model.ref_energy,
                final_energy.item() - model.ref_energy,
            )
            writer.add_scalar("haar/energy/state", final_energy, data["haar"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("haar/energy/target", target_energy, data["haar"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("haar/error/state", final_energy - model.ref_energy, data["haar"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("haar/error/target", target_energy - model.ref_energy, data["haar"]["global"])  # type: ignore[no-untyped-call]
            logging.info("Displaying the largest amplitudes")
            indices = target_psi.abs().argsort(descending=True)
            text = []
            for index in indices[: self.logging_psi]:
                this_config = model.show_config(configs[index])
                logging.info(
                    "Configuration: %s, Target amplitude: %s, Final amplitude: %s",
                    this_config,
                    f"{target_psi[index].item():.8f}",
                    f"{psi[index].item():.8f}",
                )
                text.append(
                    f"Configuration: {this_config}, Target amplitude: {target_psi[index].item():.8f}, Final amplitude: {psi[index].item():.8f}"
                )
            writer.add_text("config", "\n".join(text), data["haar"]["global"])  # type: ignore[no-untyped-call]
            writer.flush()  # type: ignore[no-untyped-call]

            logging.info("Saving model checkpoint")
            data["haar"]["pool"] = (configs, original_psi)
            data["haar"]["global"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            ctx.save(data, data["haar"]["global"])
            logging.info("Checkpoint successfully saved")

            logging.info("Current optimization cycle completed")


subcommand_dict["haar"] = HaarConfig
subcommand_dict["imag"] = HaarConfig
