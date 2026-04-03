"""
PySCF-compatible FCI solver plugin for QMP's HAAR algorithm.

This module provides a HAAR class that conforms to PySCF's FCI solver interface,
allowing it to be used as a drop-in replacement for PySCF's native FCI solver
in CASCI/CASSCF calculations.

Example
-------
>>> from pyscf import gto, scf, mcscf
>>> from qmp.plugins.pyscf import HAAR, HAARSCF
>>> mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
>>> mf = scf.RHF(mol).run()
>>> mc = HAARSCF(mf, 2, 2)
>>> mc.kernel()
"""

from __future__ import annotations

import copy
import logging
import sys
import typing
import dataclasses

import numpy
import torch

# PySCF imports (optional, required for CASSCF integration)
try:
    import pyscf.lib
    from pyscf import mcscf
    from pyscf.lib import logger
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False
    pyscf = None  # type: ignore[misc,assignment]

# QMP internal imports
from ..models.pyscf import Model, ModelConfig
from ..utility.model_dict import NetworkProto


class HAAR(pyscf.lib.StreamObject if HAS_PYSCF else object):  # type: ignore[misc,assignment]
    """HAAR FCI solver for PySCF.

    This solver uses the Hamiltonian-Guided Autoregressive (HAAR) algorithm
    to solve the FCI problem. It is designed to be used as a drop-in
    replacement for PySCF's native FCI solver in CASCI/CASSCF calculations.

    The wave function is represented as a set of configurations (determinants)
    and their amplitudes, which can be converted to/from PySCF's CI vector
    format for RDM calculations.

    Attributes
    ----------
    mol : Mole object or None
        PySCF molecule object.
    sampling_count : int
        Number of configurations to sample from the neural network.
    krylov_iteration : int
        Number of Krylov iterations for the Lanczos algorithm.
    krylov_extend_count : int
        Number of configurations to add during Krylov subspace extension.
    krylov_threshold : float
        Convergence threshold for Krylov iteration.
    local_step : int
        Number of local optimization steps.
    local_loss : float
        Loss threshold for local optimization convergence.
    network_name : str
        Name of the neural network architecture (e.g., "mlp/u1u1").
    network_hidden : tuple
        Hidden layer sizes for the neural network.
    learning_rate : float
        Learning rate for the optimizer.
    max_cycle : int
        Maximum number of global optimization cycles.
    device : str or torch.device
        Device to run on ('cpu' or 'cuda').
    dtype : str or torch.dtype
        Data type for the network.
    nroots : int
        Number of roots to solve for.
    verbose : int
        Verbosity level (0-10).

    Examples
    --------
    >>> from pyscf import gto, scf, mcscf
    >>> from qmp.plugins.pyscf import HAAR
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASSCF(mf, 2, 2)
    >>> mc.fcisolver = HAAR(mol)
    >>> mc.fcisolver.sampling_count = 512
    >>> mc.kernel()
    """

    def __init__(self, mol: typing.Any = None, **kwargs: typing.Any) -> None:
        """Initialize the HAAR solver.

        Parameters
        ----------
        mol : Mole object, optional
            PySCF molecule object. If None, some attributes must be set manually.
        **kwargs : dict
            Additional keyword arguments to override default parameters.
        """
        self.mol = mol

        # Set up logging
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = 0
        else:
            self.stdout = mol.stdout  # type: ignore[union-attr]
            self.verbose = mol.verbose  # type: ignore[union-attr]

        # HAAR algorithm parameters
        self.sampling_count: int = 1024
        self.sampling_count_last: int = 1024
        self.krylov_iteration: int = 32
        self.krylov_extend_count: int = 64
        self.krylov_threshold: float = 1e-8
        self.krylov_single_extend: bool = False
        self.krylov_extend_first: bool = False
        self.local_step: int = 200
        self.local_loss: float = 1e-8
        self.max_cycle: int = 10

        # Network parameters
        self.network_name: str = "mlp/u1u1"
        self.network_hidden: tuple[int, ...] = (512, 512)

        # Optimizer parameters
        self.learning_rate: float = 1e-3
        self.optimizer_name: str = "Adam"

        # Runtime parameters
        self.device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype: str | torch.dtype = "float64"
        self.random_seed: int | None = None

        # Multi-root support
        self.nroots: int = 1
        self.spin: int | None = None  # Spin multiplicity - 1 (2S)

        # Internal state for storing results
        self._converged: bool = False
        self._energy: float | None = None
        self._nuclear_repulsion: float = 0.0
        self._configs: torch.Tensor | None = None
        self._psi: torch.Tensor | None = None
        self._norb: int | None = None
        self._nelec: tuple[int, int] | None = None
        self._network: NetworkProto | None = None
        self._model: Model | None = None

        # Update with any provided kwargs
        self.__dict__.update(kwargs)

        # Keys for PySCF's StreamObject
        if HAS_PYSCF:
            self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose: int | None = None) -> "HAAR":
        """Print the solver parameters."""
        if not HAS_PYSCF:
            return self

        log = logger.new_logger(self, verbose)
        log.info("")
        log.info("******** HAAR FCI solver flags ********")
        log.info("sampling_count         = %d", self.sampling_count)
        log.info("sampling_count_last    = %d", self.sampling_count_last)
        log.info("krylov_iteration       = %d", self.krylov_iteration)
        log.info("krylov_extend_count    = %d", self.krylov_extend_count)
        log.info("krylov_threshold       = %g", self.krylov_threshold)
        log.info("krylov_single_extend   = %s", self.krylov_single_extend)
        log.info("krylov_extend_first    = %s", self.krylov_extend_first)
        log.info("local_step             = %d", self.local_step)
        log.info("local_loss             = %g", self.local_loss)
        log.info("max_cycle              = %d", self.max_cycle)
        log.info("network_name           = %s", self.network_name)
        log.info("network_hidden         = %s", self.network_hidden)
        log.info("learning_rate          = %g", self.learning_rate)
        log.info("device                 = %s", self.device)
        log.info("dtype                  = %s", self.dtype)
        log.info("nroots                 = %d", self.nroots)
        log.info("")
        return self

    def kernel(
        self,
        h1e: numpy.ndarray,
        eri: numpy.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        ci0: typing.Any = None,
        ecore: float = 0.0,
        **kwargs: typing.Any,
    ) -> tuple[float, int]:
        """Solve the FCI problem using the HAAR algorithm.

        Parameters
        ----------
        h1e : ndarray
            1-electron integrals in MO basis, shape (norb, norb).
            In chemist's notation, same convention as PySCF.
        eri : ndarray
            2-electron repulsion integrals in MO basis.
            Shape (norb, norb, norb, norb) or compressed 4-fold symmetric.
            In chemist's (ij|kl) notation.
        norb : int
            Number of spatial orbitals.
        nelec : int or tuple[int, int]
            Number of electrons. If int, split evenly between alpha and beta.
            If tuple, (nalpha, nbeta).
        ci0 : optional
            Initial guess. Can be:
            - None: Start from scratch
            - dict: HAAR checkpoint data with 'pool' key
            - ndarray: PySCF CI vector (will be converted)
        ecore : float
            Core energy (nuclear repulsion, frozen core contribution, etc.).
        **kwargs : dict
            Additional keyword arguments. Recognized: 'orbsym'.

        Returns
        -------
        energy : float
            Ground state energy (including ecore).
        ci : int
            CI vector identifier. For nroots=1, returns 0.
        """
        # Dump flags
        self.dump_flags(self.verbose)

        # Resolve electron counts
        if isinstance(nelec, (int, numpy.integer)):
            nalpha = (int(nelec) + 1) // 2
            nbeta = int(nelec) - nalpha
        else:
            nalpha, nbeta = int(nelec[0]), int(nelec[1])

        n_electron = nalpha + nbeta
        n_spin = nalpha - nbeta

        if HAS_PYSCF and self.verbose >= logger.INFO:
            logger.info(self, "HAAR kernel: norb=%d, nalpha=%d, nbeta=%d",
                        norb, nalpha, nbeta)

        # Store parameters
        self._norb = norb
        self._nelec = (nalpha, nbeta)
        self._nuclear_repulsion = ecore

        # Build the QMP model from integrals
        if HAS_PYSCF and self.verbose >= logger.INFO:
            logger.info(self, "Building QMP model from integrals")

        model = Model(ModelConfig(
            h1e=numpy.asarray(h1e, dtype=numpy.float64),
            eri=numpy.asarray(eri, dtype=numpy.float64),
            n_orbit=norb,
            n_electron=n_electron,
            n_spin=n_spin,
            nuclear_repulsion=ecore,
        ))
        self._model = model

        # Set up device and dtype
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        if isinstance(self.dtype, str):
            dtype_map = {
                "float64": torch.float64,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }
            dtype = dtype_map.get(self.dtype, torch.float64)
        else:
            dtype = self.dtype

        # Create network
        if HAS_PYSCF and self.verbose >= logger.INFO:
            logger.info(self, "Creating network: %s", self.network_name)

        network_config = self._get_network_config()
        network = network_config.create(model)
        network = network.to(device=device, dtype=dtype)

        # Handle initial guess
        if ci0 is not None:
            if HAS_PYSCF and self.verbose >= logger.INFO:
                logger.info(self, "Loading initial guess")
            if isinstance(ci0, dict) and "haar" in ci0:
                # HAAR checkpoint data
                pool = ci0["haar"].get("pool")
                if pool is not None and pool[0] is not None:
                    configs, psi = pool
                    # Use these as starting point
            elif isinstance(ci0, numpy.ndarray):
                # PySCF CI vector - convert to pool
                configs, psi = self._ci_vector_to_configs(ci0, norb, nalpha, nbeta)
                if configs is not None and psi is not None:
                    psi = psi.to(device=device)

        # Compile network for better performance
        if HAS_PYSCF and self.verbose >= logger.INFO:
            logger.info(self, "Compiling network with torch.jit.script")
        network = torch.jit.script(network)  # type: ignore[assignment]
        self._network = network

        # Set random seed if specified
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        # Create optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)

        # Run HAAR optimization
        energy, configs, psi = self._run_haar_optimization(
            model=model,
            network=network,
            optimizer=optimizer,
            device=device,
        )

        # Add nuclear repulsion to get total energy
        total_energy = float(energy) + ecore

        # Store results
        self._converged = True
        self._energy = total_energy
        self._configs = configs
        self._psi = psi

        if HAS_PYSCF and self.verbose >= logger.INFO:
            logger.info(self, "HAAR kernel finished. Energy = %.10f", self._energy)

        return self._energy, 0

    def _get_network_config(self) -> typing.Any:
        """Get the network configuration based on network_name."""
        if self.network_name == "mlp/u1u1":
            from ..models.pyscf import MlpConfig
            return MlpConfig(hidden=self.network_hidden)
        elif self.network_name == "mlp/u1":
            from ..models.pyscf import MlpElectronConfig
            return MlpElectronConfig(hidden=self.network_hidden)
        elif self.network_name == "transformers/u1u1":
            from ..models.pyscf import TransformersConfig
            return TransformersConfig()
        elif self.network_name == "transformers/u1":
            from ..models.pyscf import TransformersElectronConfig
            return TransformersElectronConfig()
        else:
            raise ValueError(f"Unknown network name: {self.network_name}")

    def _run_haar_optimization(
        self,
        model: Model,
        network: NetworkProto,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the HAAR optimization loop.

        This is a simplified version of the HAAR algorithm that performs
        global optimization cycles.

        Parameters
        ----------
        model : Model
            The QMP model with Hamiltonian.
        network : NetworkProto
            The neural network wave function.
        optimizer : Optimizer
            The optimizer for training.
        device : torch.device
            The device to run on.

        Returns
        -------
        energy : torch.Tensor
            The final energy.
        configs : torch.Tensor
            The final configurations.
        psi : torch.Tensor
            The final amplitudes.
        """
        from ..algorithms.haar import _DynamicLanczos, _sampling_from_last_iteration
        from ..algorithms.haar import _merge_pool_from_neural_network_and_pool_from_last_iteration
        from ..utility import losses

        loss_func = getattr(losses, "sum_filtered_angle_scaled_log")

        # Initialize pool from previous run if available
        pool: tuple[torch.Tensor, torch.Tensor] | None = None

        for cycle in range(self.max_cycle):
            if HAS_PYSCF and self.verbose >= logger.INFO:
                logger.info(self, "HAAR cycle %d/%d", cycle + 1, self.max_cycle)

            # Sample configurations from neural network
            configs_nn, psi_nn, _, _ = network.generate_unique(
                self.sampling_count, block_num=1
            )

            # Sample from last iteration if available
            configs_last, psi_last = _sampling_from_last_iteration(
                pool, self.sampling_count_last
            )

            # Merge configurations
            configs, original_psi = _merge_pool_from_neural_network_and_pool_from_last_iteration(
                configs_nn, psi_nn, configs_last, psi_last
            )

            if HAS_PYSCF and self.verbose >= logger.INFO:
                logger.info(self, "  Sampled %d unique configurations", len(configs))

            # Run Lanczos (no gradient needed for this part)
            target_energy: torch.Tensor
            lanczos_results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

            with torch.no_grad():
                for lanczos_results in _DynamicLanczos(
                    model=model,
                    configs=configs,
                    psi=original_psi,
                    step=self.krylov_iteration,
                    threshold=self.krylov_threshold,
                    count_extend=self.krylov_extend_count,
                    batch_count_apply_within=1,
                    single_extend=self.krylov_single_extend,
                    first_extend=self.krylov_extend_first,
                ).run():
                    target_energy, configs, original_psi = lanczos_results[0]

            if HAS_PYSCF and self.verbose >= logger.INFO:
                logger.info(self, "  Lanczos energy: %.10f", target_energy.item())

            # Store the Lanczos eigenvector for final energy calculation
            lanczos_psi = original_psi.clone()

            # Compute target psi for optimization
            target_prob = torch.zeros_like(original_psi, dtype=torch.float64)
            for _, _, p in lanczos_results:
                target_prob += (p.conj() * p).real
            original_psi = target_prob.sqrt().to(dtype=torch.complex128)

            max_index = original_psi.abs().argmax()
            target_psi = original_psi / original_psi[max_index]

            # Local optimization with gradient enabled
            total_size = len(configs)

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                total_loss = 0.0
                total_psi = []
                # Process all configs
                psi_all = network(configs)
                psi_max = psi_all[max_index]
                psi_all = psi_all / psi_max
                loss = loss_func(psi_all, target_psi)
                loss.backward()  # type: ignore[no-untyped-call]
                total_loss_tensor = torch.tensor(loss.item())
                total_loss_tensor.psi = psi_all.detach()  # type: ignore[attr-defined]
                return total_loss_tensor

            # Run local optimization steps with gradient enabled
            with torch.enable_grad():
                for step in range(self.local_step):
                    loss = optimizer.step(closure)  # type: ignore[arg-type]
                    if step % 50 == 0 and HAS_PYSCF and self.verbose >= logger.INFO:
                        logger.info(self, "  Local step %d, loss = %.10f", step, loss.item())
                    if loss < self.local_loss:
                        break

            # Update pool for next iteration
            pool = (configs, lanczos_psi)

        # Final energy calculation using Lanczos eigenvector
        final_energy = ((lanczos_psi.conj() @ model.apply_within(configs, lanczos_psi, configs)) / (lanczos_psi.conj() @ lanczos_psi)).real

        if HAS_PYSCF and self.verbose >= logger.INFO:
            logger.info(self, "Final energy: %.10f", final_energy.item())

        return final_energy, configs, lanczos_psi

    def _configs_to_ci_vector(
        self,
        configs: torch.Tensor,
        psi: torch.Tensor,
        norb: int,
        nalpha: int,
        nbeta: int,
    ) -> numpy.ndarray:
        """Convert configs/psi representation to PySCF CI vector.

        Parameters
        ----------
        configs : torch.Tensor
            Shape (N, n_qubytes), bit-packed configurations.
        psi : torch.Tensor
            Shape (N,), amplitudes.
        norb : int
            Number of spatial orbitals.
        nalpha, nbeta : int
            Number of alpha and beta electrons.

        Returns
        -------
        ndarray
            CI vector of shape (na, nb).
        """
        from pyscf.fci import cistring

        na = cistring.num_strings(norb, nalpha)
        nb = cistring.num_strings(norb, nbeta)

        ci = numpy.zeros((na, nb), dtype=numpy.complex128)

        # Unpack bits: QMP order is little-endian bit-packed bytes
        configs_np = configs.cpu().numpy()
        psi_np = psi.cpu().numpy()
        bits = numpy.unpackbits(configs_np, axis=1, bitorder="little")

        # Extract alpha and beta occupancy (alpha=even, beta=odd bits)
        alpha_bits = bits[:, 0 : 2 * norb : 2]
        beta_bits = bits[:, 1 : 2 * norb : 2]

        # Convert to integer bitmasks
        pw2 = 2 ** numpy.arange(norb, dtype=numpy.uint64)
        alpha_dets = (alpha_bits.astype(numpy.uint64) * pw2).sum(axis=1)
        beta_dets = (beta_bits.astype(numpy.uint64) * pw2).sum(axis=1)

        # Map bitmasks to PySCF determinant indices
        try:
            addr_a = cistring.str2addr(norb, nalpha, alpha_dets)
            addr_b = cistring.str2addr(norb, nbeta, beta_dets)
        except Exception:
            # Fallback for older PySCF or if vectorized call fails
            addr_a = numpy.array([cistring.str2addr(norb, nalpha, int(d)) for d in alpha_dets])
            addr_b = numpy.array([cistring.str2addr(norb, nbeta, int(d)) for d in beta_dets])

        ci[addr_a, addr_b] = psi_np

        return ci

    def _ci_vector_to_configs(
        self,
        ci: numpy.ndarray,
        norb: int,
        nalpha: int,
        nbeta: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Convert PySCF CI vector to configs/psi representation.

        Parameters
        ----------
        ci : ndarray
            CI vector of shape (na, nb).
        norb : int
            Number of spatial orbitals.
        nalpha, nbeta : int
            Number of alpha and beta electrons.

        Returns
        -------
        configs : torch.Tensor or None
            Bit-packed configurations.
        psi : torch.Tensor or None
            Amplitudes.
        """
        from pyscf.fci import cistring

        idx_a, idx_b = numpy.where(ci != 0)
        amplitudes = ci[idx_a, idx_b]

        str_a = cistring.addrs2str(norb, nalpha, idx_a)
        str_b = cistring.addrs2str(norb, nbeta, idx_b)

        n_qubytes = (2 * norb + 7) // 8
        N = len(str_a)
        configs = numpy.zeros((N, n_qubytes), dtype=numpy.uint8)

        for j in range(norb):
            a_bit = (str_a >> j) & 1
            b_bit = (str_b >> j) & 1
            pos_a = 2 * j
            pos_b = 2 * j + 1
            configs[:, pos_a // 8] |= (a_bit << (pos_a % 8)).astype(numpy.uint8)
            configs[:, pos_b // 8] |= (b_bit << (pos_b % 8)).astype(numpy.uint8)

        return torch.from_numpy(configs), torch.from_numpy(amplitudes)

    def make_rdm1(
        self,
        state: int,
        norb: int,
        nelec: int | tuple[int, int],
        link_index: typing.Any = None,
        **kwargs: typing.Any,
    ) -> numpy.ndarray:
        """Compute the 1-particle reduced density matrix.

        Parameters
        ----------
        state : int
            State index (0 for ground state).
        norb : int
            Number of spatial orbitals.
        nelec : int or tuple[int, int]
            Number of electrons.

        Returns
        -------
        ndarray
            1-RDM, shape (norb, norb).
        """
        return self.make_rdm12(state, norb, nelec, link_index, **kwargs)[0]

    def make_rdm12(
        self,
        state: int,
        norb: int,
        nelec: int | tuple[int, int],
        link_index: typing.Any = None,
        **kwargs: typing.Any,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Compute 1- and 2-particle reduced density matrices.

        This implementation converts the sparse wave function to a full CI vector
        and uses PySCF's RDM calculation functions.

        Parameters
        ----------
        state : int
            State index (0 for ground state).
        norb : int
            Number of spatial orbitals.
        nelec : int or tuple[int, int]
            Number of electrons.
        link_index : optional
            Unused, kept for PySCF API compatibility.

        Returns
        -------
        dm1 : ndarray
            1-RDM, shape (norb, norb).
        dm2 : ndarray
            2-RDM, shape (norb, norb, norb, norb).
        """
        if self._configs is None or self._psi is None:
            raise RuntimeError("No wave function available. Run kernel() first.")

        if isinstance(nelec, (int, numpy.integer)):
            nalpha = (int(nelec) + 1) // 2
            nbeta = int(nelec) - nalpha
        else:
            nalpha, nbeta = int(nelec[0]), int(nelec[1])

        # Convert to CI vector
        ci = self._configs_to_ci_vector(self._configs, self._psi, norb, nalpha, nbeta)

        # Use PySCF's RDM calculation
        from pyscf.fci.direct_spin1 import make_rdm12

        # Take real part if wave function is effectively real
        if numpy.allclose(ci.imag, 0):
            ci = ci.real

        dm1, dm2 = make_rdm12(ci, norb, nelec)

        return dm1, dm2

    def make_rdm1s(
        self,
        state: int,
        norb: int,
        nelec: int | tuple[int, int],
        link_index: typing.Any = None,
        **kwargs: typing.Any,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Compute spin-separated 1-particle RDMs.

        Returns
        -------
        dm1a : ndarray
            Alpha 1-RDM, shape (norb, norb).
        dm1b : ndarray
            Beta 1-RDM, shape (norb, norb).
        """
        if isinstance(nelec, (int, numpy.integer)):
            nalpha = (int(nelec) + 1) // 2
            nbeta = int(nelec) - nalpha
        else:
            nalpha, nbeta = int(nelec[0]), int(nelec[1])

        dm1, _ = self.make_rdm12(state, norb, nelec, link_index, **kwargs)

        # For a closed-shell singlet, dm1a = dm1b = dm1 / 2
        # For more accurate spin-resolved RDMs, we need the full spin-RDM calculation
        dm1a = dm1 / 2
        dm1b = dm1 / 2

        return dm1a, dm1b

    def spin_square(
        self,
        civec: typing.Any,
        norb: int,
        nelec: int | tuple[int, int],
    ) -> tuple[float, float]:
        """Compute <S^2> expectation value.

        For U(1)×U(1) symmetric wave functions (conserved nalpha and nbeta),
        S_z = (nalpha - nbeta) / 2 is fixed. The expectation value of S^2
        is computed as S_z(S_z + 1) for the spin-adapted case.

        Parameters
        ----------
        civec : int or list
            State index or list of indices (unused for U(1)×U(1) case).
        norb : int
            Number of spatial orbitals (unused).
        nelec : int or tuple[int, int]
            Number of electrons.

        Returns
        -------
        ss : float
            <S^2> expectation value.
        s : float
            2S + 1 (spin multiplicity).
        """
        if isinstance(nelec, (int, numpy.integer)):
            nelecb = int(nelec) // 2
            neleca = int(nelec) - nelecb
        else:
            neleca, nelecb = int(nelec[0]), int(nelec[1])

        s = (neleca - nelecb) * 0.5
        ss = s * (s + 1)

        if isinstance(civec, int):
            return float(ss), float(s * 2 + 1)
        else:
            # Multiple states
            return [float(ss)] * len(civec), [float(s * 2 + 1)] * len(civec)

    @property
    def nstates(self) -> int:
        """Number of states (alias for nroots)."""
        return self.nroots

    @property
    def e_tot(self) -> float:
        """Total energy of the ground state."""
        if self._energy is None:
            raise RuntimeError("No energy available. Run kernel() first.")
        return self._energy


def HAARSCF(
    mf: typing.Any,
    norb: int,
    nelec: int | tuple[int, int],
    **kwargs: typing.Any,
) -> typing.Any:
    """Create a CASSCF object using the HAAR FCI solver.

    This is a convenience function that creates a CASSCF object and sets
    the FCI solver to HAAR.

    Parameters
    ----------
    mf : SCF object
        Mean-field object from PySCF (RHF, ROHF, UHF, etc.).
    norb : int
        Number of active orbitals.
    nelec : int or tuple[int, int]
        Number of active electrons.
    **kwargs : dict
        Additional keyword arguments passed to HAAR solver.

    Returns
    -------
    CASSCF
        CASSCF object with HAAR as the FCI solver.

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from qmp.plugins.pyscf import HAARSCF
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
    >>> mf = scf.RHF(mol).run()
    >>> mc = HAARSCF(mf, 2, 2)
    >>> mc.fcisolver.sampling_count = 512
    >>> mc.kernel()
    """
    if not HAS_PYSCF:
        raise ImportError("PySCF is required for HAARSCF")

    mc = mcscf.CASSCF(mf, norb, nelec)
    mc.fcisolver = HAAR(mf.mol, **kwargs)
    return mc


def HAARCI(
    mf: typing.Any,
    norb: int,
    nelec: int | tuple[int, int],
    **kwargs: typing.Any,
) -> typing.Any:
    """Create a CASCI object using the HAAR FCI solver.

    This is a convenience function that creates a CASCI object and sets
    the FCI solver to HAAR.

    Parameters
    ----------
    mf : SCF object
        Mean-field object from PySCF.
    norb : int
        Number of active orbitals.
    nelec : int or tuple[int, int]
        Number of active electrons.
    **kwargs : dict
        Additional keyword arguments passed to HAAR solver.

    Returns
    -------
    CASCI
        CASCI object with HAAR as the FCI solver.
    """
    if not HAS_PYSCF:
        raise ImportError("PySCF is required for HAARCI")

    mc = mcscf.CASCI(mf, norb, nelec)
    mc.fcisolver = HAAR(mf.mol, **kwargs)
    return mc