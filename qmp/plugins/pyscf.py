"""
PySCF-compatible FCI solver plugin for QMP.

This module exposes a :class:`Solver` that conforms to the PySCF FCI-solver
interface::

    solver = Solver(action_name="haar", max_absolute_step=10, ...)
    energy, ci = solver.kernel(h1e, eri, norb, nelec)

Internally the solver:

1. Builds a :class:`~qmp.models.pyscf.Model` from the supplied integral arrays.
2. Creates a temporary output directory (for TensorBoard logs).
3. Constructs a :class:`_PyscfRuntimeContext` that bypasses Hydra and replaces
   ``sys.exit`` with a Python exception so that the kernel can *return* the result
   instead of terminating the process.
4. Imports and instantiates the requested QMP algorithm (e.g. ``haar``, ``vmc``)
   and calls its ``main`` method.
5. After the algorithm reaches ``max_absolute_step``, computes the variational
   energy from the trained network and returns ``(energy, network_state_dict)``.
   The returned ``network_state_dict`` may be passed back as ``ci0`` to warm-start
   subsequent calls.
"""

import importlib
import logging
import pathlib
import tempfile
import typing

import dacite
import numpy
import omegaconf
import torch

from ..models.pyscf import Model, ModelConfig
from ..utility.action_dict import action_dict
from ..utility.context import DACITE_CAST, RuntimeContext
from ..utility.model_dict import ModelProto, NetworkProto


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _AlgorithmComplete(Exception):
    """
    Raised by :class:`_PyscfRuntimeContext` when ``max_absolute_step`` is
    reached, so that the solver can recover the final checkpoint data and
    return it to the caller instead of terminating the process.
    """

    def __init__(self, data: dict[str, typing.Any]) -> None:
        self.data = data


class _PyscfRuntimeContext(RuntimeContext):
    """
    A :class:`~qmp.utility.context.RuntimeContext` subclass tailored for
    programmatic (non-Hydra) use from the PySCF plugin.

    Key differences from the base class:

    * ``folder()`` returns an explicit temporary directory instead of reading
      the Hydra output path.
    * ``setup()`` creates the folder and configures random state without
      requiring Hydra to be initialised.
    * ``create_model()`` ignores the config argument and returns the
      pre-built :class:`~qmp.models.pyscf.Model` that was injected at
      construction time.
    * ``create_network()`` delegates to the base implementation and also
      saves a reference to the created network so the solver can evaluate
      the final energy.
    * ``save()`` raises :class:`_AlgorithmComplete` instead of calling
      ``sys.exit(0)`` when ``max_absolute_step`` is reached, enabling the
      caller to retrieve the results and return normally.
    """

    def __init__(
        self,
        model_instance: ModelProto,
        folder_path: pathlib.Path,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_instance: ModelProto = model_instance
        self._folder_path: pathlib.Path = folder_path
        self._network: NetworkProto | None = None

    # -- Hydra bypass -----------------------------------------------------------

    def folder(self) -> pathlib.Path:
        return self._folder_path

    def setup(self) -> dict[str, typing.Any]:
        self._folder_path.mkdir(parents=True, exist_ok=True)
        logging.info("Log directory: %s", self._folder_path)

        logging.info("Disabling PyTorch's default gradient computation")
        torch.set_grad_enabled(False)

        if self.random_seed is not None:
            logging.info("Setting random seed to: %d", self.random_seed)
            torch.manual_seed(self.random_seed)
        else:
            current_seed = torch.seed()
            logging.info("Random seed not specified, using current seed: %d", current_seed)

        return {}

    # -- Model injection --------------------------------------------------------

    def create_model(self, model_config: omegaconf.DictConfig) -> ModelProto:
        return self._model_instance

    # -- Network reference capture ----------------------------------------------

    def create_network(
        self,
        network_config: omegaconf.DictConfig,
        model: ModelProto,
        state_dict: dict[str, typing.Any] | None = None,
    ) -> NetworkProto:
        network = super().create_network(network_config, model, state_dict)
        self._network = network
        return network

    # -- Stop via exception instead of sys.exit ---------------------------------

    def save(self, data: dict[str, typing.Any], step: int) -> None:
        if self.max_relative_step is not None:
            self.max_absolute_step = step + self.max_relative_step - 1
            self.max_relative_step = None
        if step == self.max_absolute_step:
            logging.info("Reached the maximum step, returning from solver")
            raise _AlgorithmComplete(data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Solver:
    """
    PySCF-compatible FCI solver backed by QMP algorithms.

    The interface mirrors that of ``pyscf.fci.direct_spin1.FCISolver`` so that
    this solver can be used as a drop-in replacement inside PySCF workflows::

        from qmp.plugins.pyscf import Solver
        solver = Solver(action_name="haar", max_absolute_step=10)
        energy, ci = solver.kernel(h1e, eri, norb, nelec)

    Parameters
    ----------
    action_name : str
        Name of the QMP algorithm to run (e.g. ``"haar"`` or ``"vmc"``).
        The corresponding module ``qmp.algorithms.<action_name>`` is imported
        automatically.
    action_params : dict, optional
        Keyword arguments forwarded to the algorithm's configuration dataclass.
    network_name : str
        Name of the network architecture registered in the model's
        ``network_dict`` (e.g. ``"mlp/u1u1"``).
    network_params : dict, optional
        Keyword arguments forwarded to the network configuration dataclass.
    optimizer_name : str
        Name of a ``torch.optim`` optimizer class (e.g. ``"Adam"``).
    optimizer_params : dict, optional
        Keyword arguments forwarded to the optimizer constructor
        (e.g. ``{"lr": 1e-3}``).
    max_absolute_step : int
        Number of global algorithm steps to execute before returning.
        This parameter is required when using the solver programmatically
        (unlike the CLI which can run indefinitely).
    device : str or torch.device
        Compute device (default: ``"cpu"``).
    dtype : str or torch.dtype, optional
        Floating-point dtype for the network parameters (e.g. ``"float64"``).
    random_seed : int, optional
        Random seed for reproducibility.
    sampling_count : int
        Number of configurations to sample when computing the final
        variational energy after the algorithm finishes (default: 4096).
    """

    def __init__(
        self,
        *,
        action_name: str = "haar",
        action_params: dict[str, typing.Any] | None = None,
        network_name: str = "mlp/u1u1",
        network_params: dict[str, typing.Any] | None = None,
        optimizer_name: str = "Adam",
        optimizer_params: dict[str, typing.Any] | None = None,
        max_absolute_step: int = 10,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = None,
        random_seed: int | None = None,
        sampling_count: int = 4096,
    ) -> None:
        self.action_name = action_name
        self.action_params: dict[str, typing.Any] = action_params or {}
        self.network_name = network_name
        self.network_params: dict[str, typing.Any] = network_params or {}
        self.optimizer_name = optimizer_name
        self.optimizer_params: dict[str, typing.Any] = optimizer_params or {"lr": 1e-3}
        self.max_absolute_step = max_absolute_step
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.random_seed = random_seed
        self.sampling_count = sampling_count

    def kernel(
        self,
        h1: numpy.ndarray,
        eri: numpy.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        ci0: dict[str, torch.Tensor] | None = None,
        nuclear_repulsion: float = 0.0,
        ref_energy: float = 0.0,
        **kwargs: typing.Any,
    ) -> tuple[float, dict[str, torch.Tensor]]:
        """
        Solve the FCI problem for the given integrals.

        Parameters
        ----------
        h1 : numpy.ndarray
            1-electron integrals in MO basis, shape ``(norb, norb)``, in
            chemist's notation (same convention as PySCF).
        eri : numpy.ndarray
            2-electron repulsion integrals in MO basis in chemist's (ij|kl)
            notation.  Accepted shapes:

            * ``(norb, norb, norb, norb)`` – full 4-index array.
            * ``(nij, nij)`` – 4-fold symmetric compressed array
              (``nij = norb*(norb+1)//2``).
        norb : int
            Number of spatial orbitals.
        nelec : int or tuple[int, int]
            Number of electrons.  If a plain ``int``, the electrons are split
            as evenly as possible between alpha and beta spin (rounding alpha
            up).  If a 2-tuple ``(nalpha, nbeta)``, each spin count is used
            directly.
        ci0 : dict[str, torch.Tensor], optional
            Network state dict from a previous :meth:`kernel` call.  When
            provided the network is warm-started from this state, analogous
            to passing an initial CI vector in a conventional FCI solver.
        nuclear_repulsion : float
            Constant energy offset added to the Hamiltonian (e.g. nuclear
            repulsion energy).  Default is 0.
        ref_energy : float
            Reference energy used for logging only (e.g. a known FCI
            result).  Default is 0.
        **kwargs
            Ignored; present for drop-in compatibility with PySCF solvers.

        Returns
        -------
        energy : float
            Variational energy estimate after ``max_absolute_step`` global
            algorithm steps.
        ci : dict[str, torch.Tensor]
            Final network state dict.  This can be passed back as ``ci0`` in
            a subsequent call to warm-start the optimisation.
        """
        # ------------------------------------------------------------------
        # 1. Resolve nelec
        # ------------------------------------------------------------------
        # Save the caller's gradient state so we can restore it on exit.
        prev_grad_enabled = torch.is_grad_enabled()

        if isinstance(nelec, int):
            nalpha = (nelec + 1) // 2
            nbeta = nelec - nalpha
        else:
            nalpha, nbeta = int(nelec[0]), int(nelec[1])

        n_electron = nalpha + nbeta
        n_spin = nalpha - nbeta

        # ------------------------------------------------------------------
        # 2. Build the PySCF model
        # ------------------------------------------------------------------
        model_config = ModelConfig(
            h1e=numpy.asarray(h1, dtype=numpy.float64),
            eri=numpy.asarray(eri, dtype=numpy.float64),
            n_orbit=norb,
            n_electron=n_electron,
            n_spin=n_spin,
            nuclear_repulsion=nuclear_repulsion,
            ref_energy=ref_energy,
        )
        model = Model(model_config)

        # ------------------------------------------------------------------
        # 3. Import the algorithm module (registers action in action_dict)
        # ------------------------------------------------------------------
        importlib.import_module(f"qmp.algorithms.{self.action_name}")

        # ------------------------------------------------------------------
        # 4. Build the runtime config (OmegaConf DictConfig)
        # ------------------------------------------------------------------
        runtime_config = omegaconf.OmegaConf.create(
            {
                # "model" entry is required by the algorithm's main() signature
                # but is ignored because _PyscfRuntimeContext.create_model()
                # returns the pre-built model.
                "model": {"name": "pyscf", "params": {}},
                "network": {"name": self.network_name, "params": self.network_params},
                "optimizer": {"name": self.optimizer_name, "params": self.optimizer_params},
            }
        )

        # ------------------------------------------------------------------
        # 5. Create the runtime context
        # ------------------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmpdir:
            context = _PyscfRuntimeContext(
                model_instance=model,
                folder_path=pathlib.Path(tmpdir),
                device=self.device,
                dtype=self.dtype,
                random_seed=self.random_seed,
                max_absolute_step=self.max_absolute_step,
            )

            # setup() creates the folder and sets the grad mode; it returns
            # the initial checkpoint dict (empty when no ci0 is provided).
            checkpoint_data: dict[str, typing.Any] = context.setup()

            # If ci0 is supplied, warm-start the network from the saved state.
            if ci0 is not None:
                checkpoint_data["network"] = ci0

            # ------------------------------------------------------------------
            # 6. Instantiate the algorithm
            # ------------------------------------------------------------------
            run = dacite.from_dict(
                data_class=action_dict[self.action_name],
                data=self.action_params,
                config=dacite.Config(cast=DACITE_CAST),
            )

            # ------------------------------------------------------------------
            # 7. Run the algorithm; catch the completion signal
            # ------------------------------------------------------------------
            final_data: dict[str, typing.Any] = {}
            try:
                run.main(
                    context=context,
                    runtime_config=runtime_config,
                    checkpoint_data=checkpoint_data,
                )
                # If main() returns normally (no max_absolute_step set, or the
                # algorithm exits for another reason), use whatever data is
                # available.
            except _AlgorithmComplete as exc:
                final_data = exc.data

            # ------------------------------------------------------------------
            # 8. Compute variational energy from the trained network
            # ------------------------------------------------------------------
            network = context._network
            if network is None:
                raise RuntimeError(
                    "The algorithm did not create a network. "
                    "Ensure that the selected action calls context.create_network()."
                )

            # Load the final network weights (the algorithm may have updated
            # them after context._network was first set).
            final_state_dict: dict[str, torch.Tensor] = final_data.get("network", {})
            if final_state_dict:
                network.load_state_dict(final_state_dict)

            with torch.no_grad():
                configs, psi, _, _ = network.generate_unique(self.sampling_count)
                h_psi = model.apply_within(configs, psi, configs)
                energy_tensor = (psi.conj() @ h_psi) / (psi.conj() @ psi)
                energy = energy_tensor.real.item()

        # Restore the caller's gradient state.
        torch.set_grad_enabled(prev_grad_enabled)

        return energy, final_state_dict
