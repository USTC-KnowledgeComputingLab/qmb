"""
Hamiltonian utilities for quantum many-body calculations.

This module provides high-level wrappers around qmp's Model and Hamiltonian
classes for working with wavefunctions stored in SHCI text format.
"""

from pathlib import Path
import torch
from qmp.models.fcidump import ModelConfig, Model
from qmp.utility.bitspack import pack_int, unpack_int

__all__ = ["WaveFunction", "Hamiltonian"]

# Character to value mapping for 2-bit encoding
# 0 -> 00 (empty), a -> 01 (spin down), b -> 10 (spin up), 2 -> 11 (double occupied)
CHAR_TO_VAL = {"0": 0, "a": 1, "b": 2, "2": 3}
VAL_TO_CHAR = {0: "0", 1: "a", 2: "b", 3: "2"}


class WaveFunction:
    """
    Wavefunction representation with packed configurations and amplitudes.

    Attributes
    ----------
    config : torch.Tensor
        Packed 2D tensor of shape (batch, packed_bits) containing configurations.
        Each configuration is packed using bitspack with 2 bits per orbital.
    psi : torch.Tensor
        1D complex tensor of shape (batch) containing wavefunction amplitudes.
    orbit_number : int
        Number of orbitals (determines the unpacking dimension).
    """

    def __init__(
        self,
        config: torch.Tensor | None = None,
        psi: torch.Tensor | None = None,
        orbit_number: int | None = None,
    ) -> None:
        """
        Initialize a WaveFunction with optional pre-existing data.

        Parameters
        ----------
        config : torch.Tensor, optional
            Packed configuration tensor.
        psi : torch.Tensor, optional
            Amplitude tensor.
        orbit_number : int, optional
            Number of orbitals.
        """
        self.config = config
        self.psi = psi
        self.orbit_number = orbit_number

    def to(self, device: str | torch.device) -> "WaveFunction":
        """
        Move tensors to specified device.

        Parameters
        ----------
        device : str | torch.device
            Target device (e.g., "cuda", "cpu", torch.device("cuda:0")).

        Returns
        -------
        WaveFunction
            Self for method chaining.
        """
        if self.config is not None:
            self.config = self.config.to(device)
        if self.psi is not None:
            self.psi = self.psi.to(device)
        return self

    def load(self, path: str | Path) -> "WaveFunction":
        """
        Load wavefunction from SHCI-format text file.

        File format is space/tab-separated text:
        - Column 1: index (ignored)
        - Column 2: coefficient (complex)
        - Columns 3+: config characters ('0', 'a', 'b', '2')

        Character encoding (2 bits per orbital):
            '0' -> 00 (empty orbital)
            'a' -> 01 (one electron, spin down)
            'b' -> 10 (one electron, spin up)
            '2' -> 11 (double occupied)

        Configurations are packed using bitspack with size=2 for efficient storage.

        Parameters
        ----------
        path : str | Path
            Path to the SHCI text file.

        Returns
        -------
        WaveFunction
            Self for method chaining.
        """
        path = Path(path)
        psi_list = []
        config_list = []

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                # Column 0: index (ignored), Column 1: coefficient
                psi_list.append(complex(parts[1]))

                # Columns 2+: configuration characters
                config = [CHAR_TO_VAL[c] for c in parts[2:]]
                config_list.append(config)

        # Infer orbital count from first configuration
        self.orbit_number = len(config_list[0]) if config_list else 0

        # Use complex128 for Hamiltonian operations
        self.psi = torch.tensor(psi_list, dtype=torch.complex128)

        # Pack 2-bit values for efficient storage
        config_vals = torch.tensor(config_list, dtype=torch.uint8)
        self.config = pack_int(config_vals, size=2)

        return self

    def dump(self, path: str | Path) -> "WaveFunction":
        """
        Dump wavefunction to SHCI-format text file.

        Output format matches the load format:
        - Column 1: sequential index
        - Column 2: coefficient (real part, scientific notation)
        - Columns 3+: config characters

        Parameters
        ----------
        path : str | Path
            Path to write the output file.

        Returns
        -------
        WaveFunction
            Self for method chaining.

        Raises
        ------
        ValueError
            If wavefunction is not initialized (missing config, psi, or orbit_number).
        """
        if self.config is None or self.psi is None or self.orbit_number is None:
            raise ValueError("WaveFunction not initialized")

        path = Path(path)

        # Unpack to recover 2-bit values per orbital
        config_vals = unpack_int(self.config, size=2, last_dim=self.orbit_number)

        with open(path, "w") as f:
            for i, (coef, chars_vals) in enumerate(zip(self.psi, config_vals)):
                # Convert 2-bit values back to characters
                chars = [VAL_TO_CHAR[int(v)] for v in chars_vals]
                # Tab-separated format: index, coefficient, config characters
                line = f"{i}\t{coef.real.item():.8e}\t" + " ".join(chars) + "\n"
                f.write(line)

        return self


class Hamiltonian:
    """
    High-level Hamiltonian wrapper for quantum chemistry calculations.

    Wraps qmp's Model class to provide convenient WaveFunction-based operations.

    Attributes
    ----------
    model : Model
        The underlying qmp Model loaded from FCIDUMP file.
    """

    def __init__(self, file_name: str | Path):
        """
        Load Hamiltonian from FCIDUMP file.

        Parameters
        ----------
        file_name : str | Path
            Path to the FCIDUMP file containing molecular integrals.
        """
        self.model = Model(ModelConfig(model_path=Path(file_name)))

    def apply_within(self, source: WaveFunction, destination: WaveFunction) -> WaveFunction:
        """
        Apply Hamiltonian to source wavefunction, projecting onto destination configs.

        Computes: H|source⟩ projected onto the span of destination configurations.
        This calculates ⟨dest_i|H|source⟩ for each destination configuration.

        Parameters
        ----------
        source : WaveFunction
            Input wavefunction with configs and amplitudes.
        destination : WaveFunction
            Target wavefunction specifying the output configurations.
            Its psi values are ignored; only configs matter.

        Returns
        -------
        WaveFunction
            Wavefunction with destination configs and computed amplitudes.
        """
        assert source.config is not None and source.psi is not None and destination.config is not None
        psi = self.model.apply_within(source.config, source.psi, destination.config)
        return WaveFunction(config=destination.config, psi=psi, orbit_number=destination.orbit_number)

    def find_relative(
        self,
        source: WaveFunction,
        count_selected: int,
        exclude: WaveFunction | None = None,
    ) -> WaveFunction:
        """
        Find configurations connected to source via Hamiltonian terms.

        Returns configurations reachable by applying single Hamiltonian terms
        to source configurations. Useful for iterative wavefunction expansion.

        Parameters
        ----------
        source : WaveFunction
            Source wavefunction with configs and amplitudes.
        count_selected : int
            Maximum number of configurations to return. Top candidates are
            prioritized by estimated importance.
        exclude : WaveFunction, optional
            Configurations to exclude from results. Defaults to source configs
            (to avoid returning the same configurations).

        Returns
        -------
        WaveFunction
            Wavefunction with found configurations. psi is None since only
            configs are computed, not amplitudes.
        """
        assert source.config is not None and source.psi is not None
        configs_exclude = exclude.config if exclude is not None else None
        configs_j = self.model.find_relative(source.config, source.psi, count_selected, configs_exclude)
        return WaveFunction(
            config=configs_j, psi=torch.ones([len(configs_j)], dtype=torch.complex128), orbit_number=source.orbit_number
        )

    def list_relative(
        self,
        source: WaveFunction,
        exclude: WaveFunction | None = None,
    ) -> WaveFunction:
        """
        List all unique configurations connected to source via Hamiltonian.

        Computes all configurations reachable by applying Hamiltonian terms,
        then deduplicates and sums amplitudes from multiple connection paths.

        Parameters
        ----------
        source : WaveFunction
            Source wavefunction with configs and amplitudes.
        exclude : WaveFunction, optional
            Configurations to exclude from results. Defaults to source configs.

        Returns
        -------
        WaveFunction
            Wavefunction with unique relative configurations and their
            accumulated amplitudes (sum over all connecting paths).
        """
        assert source.config is not None and source.psi is not None
        configs_exclude = exclude.config if exclude is not None else None
        configs_j, psi_j = self.model.list_relative(source.config, source.psi, configs_exclude)
        return WaveFunction(config=configs_j, psi=psi_j, orbit_number=source.orbit_number)

    def diagonal_term(self, source: WaveFunction) -> WaveFunction:
        """
        Compute diagonal Hamiltonian terms for given configurations.

        Returns the expectation value ⟨config|H|config⟩ for each configuration,
        which is the energy contribution from terms that don't change the state.

        Parameters
        ----------
        source : WaveFunction
            Wavefunction with configs (psi is ignored).

        Returns
        -------
        WaveFunction
            Wavefunction with same configs and diagonal energies as psi.
        """
        assert source.config is not None
        psi_diag = self.model.diagonal_term(source.config)
        return WaveFunction(config=source.config, psi=psi_diag, orbit_number=source.orbit_number)
