"""
This file implements the HAAR algorithm with iterative orbital optimization.
"""

import copy
import logging
import typing
import dataclasses
import pathlib
import omegaconf
import torch
import torch.utils.tensorboard
from scipy.optimize import linear_sum_assignment
from ..utility import losses
from ..utility.context import RuntimeContext
from ..utility.action_dict import action_dict
from ..utility.model_dict import ModelProto
from ..utility.optimizer import scale_learning_rate
from ..hamiltonian import Hamiltonian

# Import components from haar and orbit
from .haar import (
    _DynamicLanczos,
    _sampling_from_last_iteration,
    _merge_pool_from_neural_network_and_pool_from_last_iteration,
)
from .orbit import NaturalOrbitCalculator, _read_fcidump_tensors
from ..models.optimized_basis import Model as OptimizedModel, ModelConfig as OptimizedModelConfig


@dataclasses.dataclass
class HaarWithOrbitConfig:
    """
    The two-step optimization process with iterative orbital optimization.
    """

    # The source FCIDUMP file for orbital optimization
    src_fcidump: pathlib.Path

    # --- HAAR Config Params (copied from haar.py) ---
    sampling_count_from_neural_network: int = 1024
    sampling_count_from_last_iteration: int = 1024
    krylov_extend_count: int = -1
    krylov_extend_first: bool = False
    krylov_single_extend: bool = False
    krylov_iteration: int = 32
    krylov_threshold: float = 1e-8
    krylov_period: int = 256
    krylov_eigen_count: int = 1
    loss_name: str = "sum_filtered_angle_scaled_log"
    local_step: int = -1
    local_loss: float = 1e-8
    logging_psi: int = 30
    local_batch_count_generation: int = 1
    local_batch_count_apply_within: int = 1
    local_batch_count_loss_function: int = 1

    def __post_init__(self) -> None:
        if self.local_step == -1:
            self.local_step = 10000
        if self.krylov_extend_count == -1:
            self.krylov_extend_count = 2048 if self.krylov_single_extend else 64

    def _optimize_orbit(
        self, context: RuntimeContext, model: ModelProto, configs: torch.Tensor, psi: torch.Tensor, step: int
    ) -> tuple[ModelProto, pathlib.Path]:
        """
        Perform orbital optimization and return a new model with the optimized basis and its path.
        """
        logging.info("Performing orbital optimization for step %d", step)

        n_orbit = typing.cast(typing.Any, model).n_qubits // 2
        calculator = NaturalOrbitCalculator(n_orbit)

        # 1. Calculate RDM
        rdm = calculator.calculate_rdm(configs, psi)

        # 2. Diagonalize RDM to get Natural Orbitals
        # eigvals are occupation numbers, U columns are NOs in old basis
        eigvals, U = torch.linalg.eigh(rdm)

        # 3. Reorder U to be closest to Identity
        # We want to maximize sum(|U_{ii}|^2)
        cost_matrix = -(U.abs() ** 2).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # row_ind[i] is matched with col_ind[i]
        # We want column row_ind[i] of permuted_U to be column col_ind[i] of U
        permuted_U = torch.zeros_like(U)
        permuted_U[:, row_ind] = U[:, col_ind]
        U = permuted_U

        # 4. Load and Transform Integrals
        (norb, nelec, nspin), e0, h1, h2 = _read_fcidump_tensors(self.src_fcidump)

        device = context.device
        h1_so = torch.zeros((2 * norb, 2 * norb), dtype=torch.complex128, device=device)
        h1_so[0::2, 0::2] = h1.to(device)
        h1_so[1::2, 1::2] = h1.to(device)

        h2_so = torch.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=torch.complex128, device=device)
        h2_so[0::2, 0::2, 0::2, 0::2] = h2.to(device)
        h2_so[0::2, 1::2, 1::2, 0::2] = h2.to(device)
        h2_so[1::2, 0::2, 0::2, 1::2] = h2.to(device)
        h2_so[1::2, 1::2, 1::2, 1::2] = h2.to(device)

        # Transform
        h1_opt = U.conj().T @ h1_so @ U

        tmp = torch.einsum("sd,pqrs->pqrd", U, h2_so)
        tmp = torch.einsum("rc,pqrd->pqcd", U, tmp)
        tmp = torch.einsum("qb,pqcd->pbcd", U.conj(), tmp)
        h2_opt = torch.einsum("pa,pbcd->abcd", U.conj(), tmp)

        # 5. Build Hamiltonian Dict and Save
        ham_dict: dict[tuple[tuple[int, int], ...], complex] = {}
        ham_dict[()] = complex(e0)

        h1_indices = torch.nonzero(torch.abs(h1_opt) > 1e-12)
        for p, q in h1_indices:
            p, q = p.item(), q.item()
            ham_dict[((p, 1), (q, 0))] = h1_opt[p, q].item()

        h2_indices = torch.nonzero(torch.abs(h2_opt) > 1e-12)
        for p, q, r, s in h2_indices:
            p, q, r, s = p.item(), q.item(), r.item(), s.item()
            key = ((p, 1), (q, 1), (r, 0), (s, 0))
            ham_dict[key] = ham_dict.get(key, 0j) + h2_opt[p, q, r, s].item()

        site, kind, coef = Hamiltonian._prepare(ham_dict)

        output_data = {
            "hamiltonian": (site.cpu(), kind.cpu(), coef.cpu()),
            "n_qubits": 2 * norb,
            "n_electrons": nelec,
            "n_spins": nspin,
            "ref_energy": 0.0,
            "rdm_eigvals": eigvals.cpu(),
            "orbit_unitary": U.cpu(),
        }

        optimized_path = context.folder() / f"optimized_basis_step_{step}.pt"
        torch.save(output_data, optimized_path)
        logging.info("Saved optimized basis to %s", optimized_path)

        new_model = OptimizedModel(OptimizedModelConfig(model_path=optimized_path))

        return new_model, optimized_path

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        data = checkpoint_data
        model: ModelProto[typing.Any]

        # Determine which model to load (initial or resumed)
        if "haar" in data and "model_path" in data["haar"]:
            path = pathlib.Path(data["haar"]["model_path"])
            if path.exists():
                logging.info("Resuming with optimized model from %s", path)
                model = OptimizedModel(OptimizedModelConfig(model_path=path))
            else:
                logging.warning("Saved model path %s not found. Falling back to initial config.", path)
                model = context.create_model(runtime_config.model)
        else:
            model = context.create_model(runtime_config.model)

        network = context.create_network(runtime_config.network, model, checkpoint_data.get("network"))

        optimizer = context.create_optimizer(
            runtime_config.optimizer, network.parameters(), checkpoint_data.get("optimizer")
        )

        logging.info(
            "Arguments Summary: "
            "Sampling Count From neural network: %d, "
            "Sampling Count From Last iteration: %d, "
            "Krylov Extend Count: %d, "
            "Krylov Extend First: %s, "
            "Krylov Single Extend: %s, "
            "Krylov Iteration: %d, "
            "Krylov Threshold: %.10f, "
            "Krylov Period: %d, "
            "Krylov Eigen Count: %d, "
            "Loss Function: %s, "
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
            self.krylov_period,
            self.krylov_eigen_count,
            self.loss_name,
            self.local_step,
            self.local_loss,
            self.logging_psi,
            self.local_batch_count_generation,
            self.local_batch_count_apply_within,
            self.local_batch_count_loss_function,
        )

        if "haar" not in data and "imag" in data:
            logging.warning("The 'imag' action is deprecated, please use 'haar' instead.")
            data["haar"] = data["imag"]
            del data["imag"]
        if "haar" not in data:
            data["haar"] = {"global": 0, "local": 0, "lanczos": 0, "pool": None, "excited": {}}
        else:
            if "excited" not in data["haar"] or not isinstance(data["haar"]["excited"], dict):
                data["haar"]["excited"] = {}
            if data["haar"]["pool"] is not None:
                pool_configs, pool_psi = data["haar"]["pool"]
                data["haar"]["pool"] = (pool_configs.to(device=context.device), pool_psi.to(device=context.device))

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=context.folder())  # type: ignore

        while True:
            logging.info("Starting a new optimization cycle")

            # --- HAAR SAMPLING & LANCZOS ---
            logging.info("Sampling configurations from neural network")
            configs_from_neural_network, psi_from_neural_network, _, _ = network.generate_unique(
                self.sampling_count_from_neural_network, self.local_batch_count_generation
            )
            logging.info("Sampling configurations from last iteration")
            configs_from_last_iteration, psi_from_last_iteration = _sampling_from_last_iteration(
                data["haar"]["pool"], self.sampling_count_from_last_iteration
            )
            logging.info("Merging configurations")
            configs, original_psi = _merge_pool_from_neural_network_and_pool_from_last_iteration(
                configs_from_neural_network,
                psi_from_neural_network,
                configs_from_last_iteration,
                psi_from_last_iteration,
            )
            logging.info("Sampling completed, unique configurations count: %d", len(configs))

            # --- ORBITAL OPTIMIZATION ---
            # Optimize orbitals using the current best available state (merged pool/network)
            # This ensures Lanczos runs in the optimized basis.
            new_model, optimized_path = self._optimize_orbit(
                context, model, configs, original_psi, data["haar"]["global"]
            )

            # Update model and persist path
            model = new_model
            data["haar"]["model_path"] = str(optimized_path)

            logging.info("Computing target for local optimization (Lanczos)")
            target_energy: torch.Tensor
            lanczos_results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

            for lanczos_results in _DynamicLanczos(
                model=model,
                configs=configs,
                psi=original_psi,
                step=self.krylov_iteration,
                threshold=self.krylov_threshold,
                count_extend=self.krylov_extend_count,
                batch_count_apply_within=self.local_batch_count_apply_within,
                single_extend=self.krylov_single_extend,
                first_extend=self.krylov_extend_first,
                eigen_count=self.krylov_eigen_count,
                period=self.krylov_period,
            ).run():
                target_energy, configs, original_psi = lanczos_results[0]
                logging.info("Current energy: %.10f, samples: %d", target_energy.item(), len(configs))
                writer.add_scalar("haar/lanczos/energy", target_energy, data["haar"]["lanczos"])  # type: ignore
                writer.add_scalar("haar/lanczos/error", target_energy - model.ref_energy, data["haar"]["lanczos"])  # type: ignore
                data["haar"]["lanczos"] += 1

            data["haar"]["excited"][data["haar"]["global"]] = lanczos_results

            # --- TARGET PSI CALCULATION ---
            # Update target_psi based on the NEW probabilities (or old ones? technically basis changed)
            # The 'configs' indices retain their meaning "closest to identity", so we assume
            # the probability distribution over 'configs' is still roughly valid for the new basis.
            # We use the summed probability of the subspace from Lanczos as the target.
            target_prob = torch.zeros_like(original_psi, dtype=torch.float64)
            for _, _, p in lanczos_results:
                target_prob += (p.conj() * p).real
            original_psi = target_prob.sqrt().to(dtype=torch.complex128)

            max_index = original_psi.abs().argmax()
            target_psi = original_psi / original_psi[max_index]

            logging.info("Local optimization target calculated")

            # --- LOCAL OPTIMIZATION ---
            loss_func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(losses, self.loss_name)

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
                        logging.warning("Loss is NaN, restoring...")
                        success = False
                        break
                    if loss < self.local_loss:
                        break
                    if abs(loss - last_loss) < self.local_loss:
                        break
                    last_loss = loss.item()
                scale_learning_rate(optimizer, 1 << try_index)
                if success:
                    if any(torch.isnan(param).any() or torch.isinf(param).any() for param in network.parameters()):
                        logging.warning("NaN detected in parameters...")
                        success = False
                if success:
                    logging.info("Local optimization process completed")
                    data["haar"]["local"] = local_step
                    break
                network.load_state_dict(state_backup)
                optimizer.load_state_dict(optimizer_backup)
                try_index = try_index + 1

            loss = typing.cast(torch.Tensor, torch.enable_grad(closure)())  # type: ignore[no-untyped-call,call-arg]
            psi: torch.Tensor = loss.psi  # type: ignore[attr-defined]
            final_energy = ((psi.conj() @ model.apply_within(configs, psi, configs)) / (psi.conj() @ psi)).real
            logging.info(
                "Loss: %.10f, Final energy: %.10f, Target energy: %.10f, Reference energy: %.10f, Final error: %.10f",
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

            data["haar"]["pool"] = (configs, original_psi)
            data["haar"]["global"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            context.save(data, data["haar"]["global"])
            logging.info("Checkpoint successfully saved")


action_dict["haar_with_orbit"] = HaarWithOrbitConfig
