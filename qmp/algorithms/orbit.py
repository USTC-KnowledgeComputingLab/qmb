"""
This file implements orbital optimization (Natural Orbitals) for quantum many-body problems.
"""

import logging
import dataclasses
import re
import gzip
import pathlib
import typing
import torch
import omegaconf
from ..hamiltonian import Hamiltonian
from ..utility.context import RuntimeContext
from ..utility.action_dict import action_dict


@dataclasses.dataclass
class OrbitConfig:
    """
    Configuration for orbital optimization.
    """

    # The source FCIDUMP file
    src_fcidump: pathlib.Path
    # The output file for the optimized basis Hamiltonian
    dst_optimized: pathlib.Path
    # The reference energy, if not provided it will be 0 or read from checkpoint
    ref_energy: float | None = None

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        # 1. Setup and data loading
        model = context.create_model(runtime_config.model)
        data = checkpoint_data

        if "haar" not in data or "pool" not in data["haar"] or data["haar"]["pool"] is None:
            raise ValueError("No pool data found in checkpoint under 'haar'. Run HAAR first.")

        configs, psi = data["haar"]["pool"]
        configs = configs.to(device=context.device)
        psi = psi.to(device=context.device)

        n_orbit = typing.cast(typing.Any, model).n_qubits // 2
        calculator = NaturalOrbitCalculator(n_orbit)

        # 2. Calculate spatial 1-RDM and Unitary transformation
        rdm_spatial = calculator.calculate_rdm(configs, psi)
        logging.info("Spatial 1-RDM calculated. Diagonalizing...")

        # Natural orbitals diagonalize the RDM
        # U_spatial is now a (n_orbit, n_orbit) matrix for spatial orbitals only
        eigvals, U_spatial = torch.linalg.eigh(rdm_spatial)

        # 3. Load original integrals (these are spatial orbital integrals from FCIDUMP)
        (norb, nelec, nspin), e0, h1, h2 = _read_fcidump_tensors(self.src_fcidump)

        device = context.device
        U_spatial = U_spatial.to(device=device, dtype=torch.complex128)
        h1 = h1.to(device=device, dtype=torch.complex128)
        h2 = h2.to(device=device, dtype=torch.complex128)

        # 4. Transform SPATIAL orbital integrals first, then expand to spin-orbitals
        # This preserves spin symmetry - same transformation for alpha and beta
        logging.info("Transforming spatial orbital integrals...")

        # Transform h1: h1' = U^† h1 U
        h1_opt_spatial = U_spatial.conj().T @ h1 @ U_spatial

        # Transform h2 using sequential contractions: h2'_{abcd} = sum U^*_{pa} U^*_{qb} h2_{pqrs} U_{rc} U_{sd}
        # This is O(N^5) but correct for 4-index tensor transformation
        tmp = torch.einsum("sd,pqrs->pqrd", U_spatial, h2)
        tmp = torch.einsum("rc,pqrd->pqcd", U_spatial, tmp)
        tmp = torch.einsum("qb,pqcd->pbcd", U_spatial.conj(), tmp)
        h2_opt_spatial = torch.einsum("pa,pbcd->abcd", U_spatial.conj(), tmp)

        # 5. Expand transformed spatial integrals to spin-orbital basis
        # For restricted calculations, alpha and beta use the same spatial orbitals
        logging.info("Expanding transformed integrals to spin-orbital basis...")
        h1_so = torch.zeros((2 * norb, 2 * norb), dtype=torch.complex128, device=device)
        h1_so[0::2, 0::2] = h1_opt_spatial  # alpha-alpha
        h1_so[1::2, 1::2] = h1_opt_spatial  # beta-beta

        h2_so = torch.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=torch.complex128, device=device)
        h2_so[0::2, 0::2, 0::2, 0::2] = h2_opt_spatial  # aaaa
        h2_so[0::2, 1::2, 1::2, 0::2] = h2_opt_spatial  # abba
        h2_so[1::2, 0::2, 0::2, 1::2] = h2_opt_spatial  # baab
        h2_so[1::2, 1::2, 1::2, 1::2] = h2_opt_spatial  # bbbb

        # Build spin-orbital unitary for storage (block-diagonal: same U for alpha and beta)
        U_spin = torch.zeros((2 * norb, 2 * norb), dtype=torch.complex128, device=device)
        U_spin[0::2, 0::2] = U_spatial  # alpha block
        U_spin[1::2, 1::2] = U_spatial  # beta block

        # 6. Build Hamiltonian dictionary
        logging.info("Building Hamiltonian dictionary...")
        ham_dict: dict[tuple[tuple[int, int], ...], complex] = {}
        ham_dict[()] = complex(e0)

        h1_indices = torch.nonzero(torch.abs(h1_so) > 1e-12)
        for p, q in h1_indices:
            p, q = p.item(), q.item()
            ham_dict[((p, 1), (q, 0))] = h1_so[p, q].item()

        h2_indices = torch.nonzero(torch.abs(h2_so) > 1e-12)
        for p, q, r, s in h2_indices:
            p, q, r, s = p.item(), q.item(), r.item(), s.item()
            key = ((p, 1), (q, 1), (r, 0), (s, 0))
            ham_dict[key] = ham_dict.get(key, 0j) + h2_so[p, q, r, s].item()

        # 7. Save results
        logging.info("Preparing and saving optimized basis model...")
        site, kind, coef = Hamiltonian._prepare(ham_dict)

        output_data = {
            "hamiltonian": (site.cpu(), kind.cpu(), coef.cpu()),
            "n_qubits": 2 * norb,
            "n_electrons": nelec,
            "n_spins": nspin,
            "ref_energy": self.ref_energy if self.ref_energy is not None else model.ref_energy,
            "rdm_eigvals": eigvals.cpu(),
            "orbit_unitary": U_spin.cpu(),  # Store the spin-orbital unitary matrix
        }

        self.dst_optimized.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_data, self.dst_optimized)
        logging.info("Successfully saved optimized basis to %s", self.dst_optimized)


def _read_fcidump_tensors(
    file_name: pathlib.Path,
) -> tuple[tuple[int, int, int], float, torch.Tensor, torch.Tensor]:
    """
    Read FCIDUMP file and return raw tensors for h1 and h2.
    """
    logging.info("Parsing FCIDUMP file: %s", file_name)
    with (
        gzip.open(file_name, "rt", encoding="utf-8")
        if file_name.name.endswith(".gz")
        else open(file_name, "rt", encoding="utf-8") as file
    ):
        n_orbit: int | None = None
        n_electron: int | None = None
        n_spin: int | None = None
        for line in file:
            data: str = line.lower()
            if (match := re.search(r"norb\s*=\s*(\d+)", data)) is not None:
                n_orbit = int(match.group(1))
            if (match := re.search(r"nelec\s*=\s*(\d+)", data)) is not None:
                n_electron = int(match.group(1))
            if (match := re.search(r"ms2\s*=\s*(\d+)", data)) is not None:
                n_spin = int(match.group(1))
            if "&end" in data:
                break
        assert n_orbit is not None
        assert n_electron is not None
        assert n_spin is not None

        energy_0: float = 0.0
        energy_1: torch.Tensor = torch.zeros([n_orbit, n_orbit], dtype=torch.float64)
        energy_2: torch.Tensor = torch.zeros([n_orbit, n_orbit, n_orbit, n_orbit], dtype=torch.float64)
        for line in file:
            pieces: list[str] = line.split()
            if not pieces:
                continue
            coefficient: float = float(pieces[0])
            sites: tuple[int, ...] = tuple(int(i) - 1 for i in pieces[1:])
            match sites:
                case (-1, -1, -1, -1):
                    energy_0 = coefficient
                case (i, j, -1, -1):
                    energy_1[i, j] = coefficient
                    energy_1[j, i] = coefficient
                case (i, j, k, l):
                    energy_2[i, j, k, l] = coefficient
                    energy_2[i, j, l, k] = coefficient
                    energy_2[j, i, k, l] = coefficient
                    energy_2[j, i, l, k] = coefficient
                    energy_2[l, k, j, i] = coefficient
                    energy_2[k, l, j, i] = coefficient
                    energy_2[l, k, i, j] = coefficient
                    energy_2[k, l, i, j] = coefficient
                case _:
                    pass

    energy_2 = energy_2.permute(0, 2, 3, 1).contiguous() / 2
    return (n_orbit, n_electron, n_spin), energy_0, energy_1, energy_2


class NaturalOrbitCalculator:
    def __init__(self, n_orbit: int):
        self.n_orbit = n_orbit

    def calculate_rdm(self, configs: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """
        Calculate the spatial 1-RDM (One-Particle Reduced Density Matrix) for restricted calculations.

        For restricted (closed-shell) systems, we compute the spatial orbital RDM by summing
        contributions from both alpha and beta spin-orbitals, ensuring spin symmetry is preserved.

        Returns a RDM of shape (n_orbit, n_orbit) for spatial orbitals only.
        """
        n_qubits = self.n_orbit * 2
        # Calculate full spin-orbital RDM first
        rdm_so = torch.zeros((n_qubits, n_qubits), dtype=torch.complex128, device=psi.device)
        logging.info("Calculating spin-orbital 1-RDM for %d spin-orbitals", n_qubits)

        for q in range(n_qubits):
            if q % 10 == 0:
                logging.info("Processing column %d/%d", q, n_qubits)
            for p in range(n_qubits):
                op = Hamiltonian({((p, 1), (q, 0)): 1.0}, kind="fermi")
                val = psi.conj() @ op.apply_within(configs, psi, configs)
                rdm_so[p, q] = val

        # Convert to spatial RDM: D_spatial = D_aa + D_bb (for restricted closed-shell)
        # Alpha spin-orbitals are at even indices (0, 2, 4, ...), beta at odd indices (1, 3, 5, ...)
        # For restricted case, we take the average: D_spatial = (D_aa + D_bb) / 2
        # Then the spatial RDM for the transformation should preserve spin symmetry
        rdm_aa = rdm_so[0::2, 0::2]  # alpha-alpha block
        rdm_bb = rdm_so[1::2, 1::2]  # beta-beta block

        # For spin-restricted case, average the two blocks
        # This ensures the resulting natural orbitals are spin-symmetric
        rdm_spatial = (rdm_aa + rdm_bb) / 2.0

        logging.info("Spatial 1-RDM calculated from spin-orbital RDM (spin-restricted)")
        return rdm_spatial


action_dict["orbit"] = OrbitConfig
