"""
PySCF-compatible FCI solver plugin for QMP.

This module exposes a :class:`Solver` that conforms to the PySCF FCI-solver
interface::

    solver = Solver(config)
    energy, ci = solver.kernel(h1e, eri, norb, nelec)

The ``config`` argument mirrors the YAML config accepted by ``qmp.__main__``::

    config = {
        "action": {"name": "haar", "params": {}},
        "network": {"name": "mlp/u1u1", "params": {}},
        "optimizer": {"name": "Adam", "params": {"lr": 1e-3}},
        "common": {
            "device": "cpu",
            "dtype": "float64",
            "max_absolute_step": 10,
        },
    }

Internally the solver closely mimics :func:`qmp.__main__.main`:

1. Imports the requested algorithm and model modules.
2. Builds a :class:`~qmp.models.pyscf.Model` from the supplied integral arrays.
3. Constructs a :class:`_PyscfRuntimeContext` that overrides
   :meth:`~qmp.utility.context.RuntimeContext.folder` to use a temporary
   directory and :meth:`~qmp.utility.context.RuntimeContext.create_model` to
   inject the pre-built model.  All other methods — including ``setup()`` and
   ``save()`` — are inherited unchanged from :class:`RuntimeContext` and work
   via normal polymorphism.
4. Runs the algorithm via its ``main()`` method.
5. When ``max_absolute_step`` is reached, the base ``save()`` writes the
   checkpoint to the temporary directory and calls ``sys.exit(0)``.  The
   solver catches the resulting :class:`SystemExit` and reads the saved
   checkpoint file to extract the energy and CI data.

The returned ``ci`` dict may be passed back as ``ci0`` to warm-start
subsequent calls.
"""

import importlib
import pathlib
import shutil
import tempfile
import typing

import dacite
import numpy
import omegaconf
import torch

from ..models.pyscf import Model, ModelConfig
from ..utility.action_dict import action_dict
from ..utility.context import DACITE_CAST, RuntimeContext
from ..utility.model_dict import ModelProto


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _PyscfRuntimeContext(RuntimeContext):
    """
    A :class:`~qmp.utility.context.RuntimeContext` subclass for programmatic
    (non-Hydra) use from the PySCF solver plugin.

    The only behavioural differences from the base class are:

    * :meth:`folder` returns an explicit path supplied at construction time
      instead of reading the Hydra output directory.
    * :meth:`create_model` returns a pre-built :class:`~qmp.models.pyscf.Model`
      instead of instantiating one from the OmegaConf config.

    All other methods, including :meth:`~RuntimeContext.setup` and
    :meth:`~RuntimeContext.save`, are inherited unchanged.  Because they call
    ``self.folder()`` internally, they automatically use the temporary
    directory via normal polymorphism.
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

    def folder(self) -> pathlib.Path:
        return self._folder_path

    def create_model(self, model_config: omegaconf.DictConfig) -> ModelProto:
        return self._model_instance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Solver:
    """
    PySCF-compatible FCI solver backed by QMP algorithms.

    The interface mirrors that of ``pyscf.fci.direct_spin1.FCISolver``::

        from qmp.plugins.pyscf import Solver
        solver = Solver({
            "action": {"name": "haar", "params": {}},
            "network": {"name": "mlp/u1u1", "params": {}},
            "optimizer": {"name": "Adam", "params": {"lr": 1e-3}},
            "common": {"device": "cpu", "max_absolute_step": 10},
        })
        energy, ci = solver.kernel(h1e, eri, norb, nelec)

    Parameters
    ----------
    config : dict or omegaconf.DictConfig
        Configuration in the same format as the ``qmp.__main__`` YAML config.
        Required top-level keys: ``action``, ``network``, ``optimizer``,
        ``common``.  The ``model`` key is ignored (the model is built
        automatically from the integral arrays passed to :meth:`kernel`).
        ``common.max_absolute_step`` controls how many global algorithm steps
        to run before returning.
    """

    def __init__(self, config: dict[str, typing.Any] | omegaconf.DictConfig) -> None:
        if isinstance(config, dict):
            self.config: omegaconf.DictConfig = omegaconf.OmegaConf.create(config)
        else:
            self.config = config

    def kernel(
        self,
        h1e: numpy.ndarray,
        eri: numpy.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        ci0: dict[str, typing.Any] | None = None,
        nuclear_repulsion: float = 0.0,
        ref_energy: float = 0.0,
        **kwargs: typing.Any,
    ) -> tuple[float, dict[str, typing.Any]]:
        """
        Solve the electronic structure problem for the given integrals.

        Parameters
        ----------
        h1e : numpy.ndarray
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
        ci0 : dict, optional
            Checkpoint data from a previous :meth:`kernel` call (the ``ci``
            value returned by that call).  When provided it is written to the
            temporary checkpoint directory so that the algorithm resumes from
            the saved state, analogous to passing an initial CI vector in a
            conventional FCI solver.
        nuclear_repulsion : float
            Constant nuclear repulsion energy added to the Hamiltonian.
        ref_energy : float
            Reference energy used for logging only.
        **kwargs
            Ignored; present for drop-in compatibility with PySCF solvers.

        Returns
        -------
        energy : float
            Ground-state energy after ``common.max_absolute_step`` global
            algorithm steps, extracted from the saved checkpoint.
        ci : dict
            Checkpoint data (without random engine state) that can be passed
            back as ``ci0`` in a subsequent call to continue optimisation.
            For the ``haar`` algorithm this contains, among other things,
            ``ci["haar"]["pool"]`` — a ``(configs, psi)`` tuple representing
            the sampled basis and its Lanczos-optimised amplitudes.
        """
        # ------------------------------------------------------------------
        # 1. Resolve electron counts
        # ------------------------------------------------------------------
        if isinstance(nelec, int):
            nalpha = (nelec + 1) // 2
            nbeta = nelec - nalpha
        else:
            nalpha, nbeta = int(nelec[0]), int(nelec[1])

        n_electron = nalpha + nbeta
        n_spin = nalpha - nbeta

        # ------------------------------------------------------------------
        # 2. Dynamic imports — mirroring __main__.py
        # ------------------------------------------------------------------
        importlib.import_module(f"qmp.algorithms.{self.config.action.name}")
        importlib.import_module("qmp.models.pyscf")

        # ------------------------------------------------------------------
        # 3. Build PySCF model from integral arrays
        # ------------------------------------------------------------------
        model = Model(
            ModelConfig(
                h1e=numpy.asarray(h1e, dtype=numpy.float64),
                eri=numpy.asarray(eri, dtype=numpy.float64),
                n_orbit=norb,
                n_electron=n_electron,
                n_spin=n_spin,
                nuclear_repulsion=nuclear_repulsion,
                ref_energy=ref_energy,
            )
        )

        # ------------------------------------------------------------------
        # 4. Build OmegaConf runtime_config — same structure as the YAML
        #    config, but with model.name = "pyscf" so that the algorithm
        #    can log the model type.  create_model() is overridden so the
        #    params field is never read.
        # ------------------------------------------------------------------
        runtime_config: omegaconf.DictConfig = omegaconf.OmegaConf.create(
            {
                "action": omegaconf.OmegaConf.to_container(self.config.action, resolve=True),
                "model": {"name": "pyscf", "params": {}},
                "network": omegaconf.OmegaConf.to_container(self.config.network, resolve=True),
                "optimizer": omegaconf.OmegaConf.to_container(self.config.optimizer, resolve=True),
            }
        )

        # ------------------------------------------------------------------
        # 5. Construct _PyscfRuntimeContext from config.common
        # ------------------------------------------------------------------
        common_data = omegaconf.OmegaConf.to_container(self.config.common, resolve=True)
        assert isinstance(common_data, dict)
        # Use dacite for type conversion (pathlib.Path, torch.device, etc.),
        # matching the same approach used in __main__.py.
        rt = dacite.from_dict(
            data_class=RuntimeContext,
            data=common_data,
            config=dacite.Config(cast=DACITE_CAST),
        )

        # ------------------------------------------------------------------
        # 6. Save caller gradient state; create temp directory
        # ------------------------------------------------------------------
        prev_grad_enabled = torch.is_grad_enabled()
        tmpdir: pathlib.Path | None = None
        try:
            tmpdir = pathlib.Path(tempfile.mkdtemp())
            # If ci0 is provided, save it as the initial checkpoint so that
            # setup() loads it automatically — the same way __main__.py
            # resumes from an existing checkpoint file.
            if ci0 is not None:
                torch.save(ci0, tmpdir / "data.pth")

            context = _PyscfRuntimeContext(
                model_instance=model,
                folder_path=tmpdir,
                parent_path=rt.parent_path,
                random_seed=rt.random_seed,
                checkpoint_interval=rt.checkpoint_interval,
                device=rt.device,
                dtype=rt.dtype,
                max_absolute_step=rt.max_absolute_step,
                max_relative_step=rt.max_relative_step,
            )

            # ------------------------------------------------------------------
            # 7. Setup context — loads checkpoint / ci0, configures RNG
            # ------------------------------------------------------------------
            checkpoint_data = context.setup()

            # ------------------------------------------------------------------
            # 8. Instantiate algorithm — mirroring __main__.py
            # ------------------------------------------------------------------
            run = dacite.from_dict(
                data_class=action_dict[self.config.action.name],
                data=omegaconf.OmegaConf.to_container(
                    self.config.action.params, resolve=True
                ),  # type: ignore[arg-type]
                config=dacite.Config(cast=DACITE_CAST),
            )

            # ------------------------------------------------------------------
            # 9. Run algorithm — mirroring __main__.py
            #    The base save() writes the checkpoint to the temp dir and
            #    then calls sys.exit(0) when max_absolute_step is reached.
            #    We catch the resulting SystemExit so kernel() can return.
            # ------------------------------------------------------------------
            try:
                run.main(
                    context=context,
                    runtime_config=runtime_config,
                    checkpoint_data=checkpoint_data,
                )
            except SystemExit:
                pass

            # ------------------------------------------------------------------
            # 10. Read the checkpoint that was saved by save()
            # ------------------------------------------------------------------
            data_path = tmpdir / "data.pth"
            if data_path.exists():
                data: dict[str, typing.Any] = torch.load(
                    data_path, map_location="cpu", weights_only=True
                )
            else:
                data = {}

            # ------------------------------------------------------------------
            # 11. Extract energy and build ci output
            # ------------------------------------------------------------------
            energy = _extract_energy(data, self.config.action.name)
            # Strip the random engine state — it is stale by the next call.
            ci: dict[str, typing.Any] = {k: v for k, v in data.items() if k != "random"}

        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)
            torch.set_grad_enabled(prev_grad_enabled)

        return energy, ci


def _extract_energy(data: dict[str, typing.Any], action_name: str) -> float:
    """
    Extract the final ground-state energy from algorithm checkpoint data.

    Parameters
    ----------
    data : dict
        Checkpoint data loaded from the temporary directory.
    action_name : str
        Name of the algorithm that produced the checkpoint.

    Returns
    -------
    float
        Ground-state energy, or ``0.0`` if it cannot be determined from the
        checkpoint (e.g. the algorithm does not store energy in its checkpoint).
    """
    if action_name in ("haar", "imag"):
        haar_data = data.get("haar")
        if haar_data is None:
            return 0.0
        haar_global_step: int = haar_data.get("global", 0)
        # In haar.py, excited states are stored with key = global_step *before*
        # the increment, so the last entry key is (global_step - 1).
        last_key = haar_global_step - 1
        excited: dict[int, list[typing.Any]] = haar_data.get("excited", {})
        results = excited.get(last_key)
        if results:
            # results is list[(energy_tensor, configs, psi)]; index 0 is ground state.
            energy_val = results[0][0]
            if hasattr(energy_val, "item"):
                return float(energy_val.item())
    return 0.0
