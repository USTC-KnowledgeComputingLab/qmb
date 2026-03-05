"""
Tests for qmp.hamiltonian.Hamiltonian.

We use a 4-site 1D spinless free-fermion Hamiltonian (n_qubytes=1):

    H = n + t
      = Σᵢ nᵢ  +  2 · Σ_{⟨i,j⟩} (c†ᵢcⱼ + c†ⱼcᵢ)

where nᵢ = c†ᵢcᵢ is the number operator (coefficient 1) and the hopping is
nearest-neighbor only with coefficient 2. The distinct coefficients make it
easy to verify that the diagonal (n) and off-diagonal (t) parts are both
computed correctly.

Operator-key convention in the Hamiltonian dict:
    Key   ((s₀, k₀), (s₁, k₁), ...)  →  operator  O_{k₀,s₀} O_{k₁,s₁}
    with  k=0: annihilation,  k=1: creation.
    The operators are applied right-to-left, i.e. the last listed operator
    acts on the state first.

Config encoding (n_qubytes=1): bit k of the uint8 value = occupation of site k.
    config = Σᵢ nᵢ · 2ⁱ  (bit 0 = site 0, …, bit 3 = site 3)

Selected basis states used in tests:
    0  = 0b0000  – vacuum
    1  = 0b0001  – site 0 occupied
    2  = 0b0010  – site 1 occupied
    4  = 0b0100  – site 2 occupied
    8  = 0b1000  – site 3 occupied
    5  = 0b0101  – sites 0,2 occupied
    6  = 0b0110  – sites 1,2 occupied
    10 = 0b1010  – sites 1,3 occupied
    15 = 0b1111  – all sites occupied
"""

import pytest
import torch

from qmp.hamiltonian import Hamiltonian

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def hamiltonian() -> Hamiltonian:
    """
    4-site 1D spinless free-fermion Hamiltonian: H = Σ nᵢ + 2·Σ_{⟨i,j⟩}(c†ᵢcⱼ + h.c.)

    Number-operator coefficient = 1, nearest-neighbor hopping coefficient = 2.
    Compiled and cached once per test session to avoid repeated JIT overhead.
    """
    return Hamiltonian(
        {
            # Number operators (coefficient 1)
            ((0, 1), (0, 0)): 1.0,  # n₀
            ((1, 1), (1, 0)): 1.0,  # n₁
            ((2, 1), (2, 0)): 1.0,  # n₂
            ((3, 1), (3, 0)): 1.0,  # n₃
            # Nearest-neighbor hopping (coefficient 2)
            ((0, 1), (1, 0)): 2.0,  # c†₀c₁
            ((1, 1), (0, 0)): 2.0,  # c†₁c₀
            ((1, 1), (2, 0)): 2.0,  # c†₁c₂
            ((2, 1), (1, 0)): 2.0,  # c†₂c₁
            ((2, 1), (3, 0)): 2.0,  # c†₂c₃
            ((3, 1), (2, 0)): 2.0,  # c†₃c₂
        },
        kind="fermi",
    )


# ---------------------------------------------------------------------------
# Config tensors (shape [batch, n_qubytes=1])
# ---------------------------------------------------------------------------

C_vacuum = torch.tensor([[0]], dtype=torch.uint8)   # |0000⟩
C_site0 = torch.tensor([[1]], dtype=torch.uint8)    # |1000⟩ – site 0 occupied
C_site1 = torch.tensor([[2]], dtype=torch.uint8)    # |0100⟩ – site 1 occupied
C_site2 = torch.tensor([[4]], dtype=torch.uint8)    # |0010⟩ – site 2 occupied
C_site3 = torch.tensor([[8]], dtype=torch.uint8)    # |0001⟩ – site 3 occupied
C_sites02 = torch.tensor([[5]], dtype=torch.uint8)  # |1010⟩ – sites 0,2 occupied
C_sites12 = torch.tensor([[6]], dtype=torch.uint8)  # |0110⟩ – sites 1,2 occupied
C_sites13 = torch.tensor([[10]], dtype=torch.uint8) # |0101⟩ – sites 1,3 occupied
C_all = torch.tensor([[15]], dtype=torch.uint8)     # |1111⟩ – all sites occupied

# All 1-particle configs (sites 0–3)
C_1particle = torch.tensor([[1], [2], [4], [8]], dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Tests for diagonal_term
# ---------------------------------------------------------------------------


class TestDiagonalTerm:
    """diagonal_term returns the diagonal matrix element H[config, config]."""

    def test_vacuum_is_zero(self, hamiltonian: Hamiltonian) -> None:
        """Vacuum has no particles; all number operators vanish."""
        result = hamiltonian.diagonal_term(C_vacuum)
        torch.testing.assert_close(result, torch.tensor([0.0 + 0j], dtype=torch.complex128))

    def test_single_particle_equals_n_coefficient(self, hamiltonian: Hamiltonian) -> None:
        """Single-particle states give diagonal = 1 (the number-operator coefficient)."""
        result = hamiltonian.diagonal_term(C_site0)
        torch.testing.assert_close(result, torch.tensor([1.0 + 0j], dtype=torch.complex128))

    def test_two_particles_sums_n_coefficients(self, hamiltonian: Hamiltonian) -> None:
        """Two non-adjacent occupied sites: diagonal = 2·1 = 2 (hopping doesn't contribute)."""
        result = hamiltonian.diagonal_term(C_sites02)
        torch.testing.assert_close(result, torch.tensor([2.0 + 0j], dtype=torch.complex128))

    def test_all_occupied_diagonal_is_four(self, hamiltonian: Hamiltonian) -> None:
        """All 4 sites occupied: diagonal = 4·1 = 4 (hopping blocked by Pauli exclusion)."""
        result = hamiltonian.diagonal_term(C_all)
        torch.testing.assert_close(result, torch.tensor([4.0 + 0j], dtype=torch.complex128))

    def test_batch_all_single_particle_states(self, hamiltonian: Hamiltonian) -> None:
        """All four 1-particle basis states give diagonal = 1 in a single batched call."""
        result = hamiltonian.diagonal_term(C_1particle)
        expected = torch.ones(4, dtype=torch.complex128)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Tests for apply_within
# ---------------------------------------------------------------------------


class TestApplyWithin:
    """apply_within(configs_i, psi_i, configs_j) computes (H·ψ) restricted to configs_j."""

    def test_n_and_t_coefficients_are_distinguishable(self, hamiltonian: Hamiltonian) -> None:
        """H|site1⟩ in the 1-particle sector gives [2, 1, 2] for configs [site0, site1, site2].

        The diagonal element is 1 (n coefficient), off-diagonal hops are 2 (t coefficient).
        """
        configs_j = torch.tensor([[1], [2], [4]], dtype=torch.uint8)  # sites 0, 1, 2
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        result = hamiltonian.apply_within(C_site1, psi_i, configs_j)
        expected = torch.tensor([2.0 + 0j, 1.0 + 0j, 2.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_diagonal_only_from_all_occupied(self, hamiltonian: Hamiltonian) -> None:
        """H|1111⟩ = 4·|1111⟩: hopping is Pauli-blocked, only n terms contribute."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        result = hamiltonian.apply_within(C_all, psi_i, C_all)
        torch.testing.assert_close(result, torch.tensor([4.0 + 0j], dtype=torch.complex128))

    def test_no_transition_to_vacuum(self, hamiltonian: Hamiltonian) -> None:
        """H never maps any configuration to the vacuum |0000⟩."""
        configs_i = torch.tensor([[0], [1], [2], [4], [8], [5], [6], [15]], dtype=torch.uint8)
        psi_i = torch.ones(8, dtype=torch.complex128)
        result = hamiltonian.apply_within(configs_i, psi_i, C_vacuum)
        torch.testing.assert_close(result, torch.tensor([0.0 + 0j], dtype=torch.complex128))

    def test_hopping_only_between_adjacent_sites(self, hamiltonian: Hamiltonian) -> None:
        """c†₂c₃ connects site3 to site2 (adjacent); site0 is not connected to site3."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        # site0 should receive 0 (non-adjacent to site3)
        result_non_adjacent = hamiltonian.apply_within(C_site3, psi_i, C_site0)
        torch.testing.assert_close(result_non_adjacent, torch.tensor([0.0 + 0j], dtype=torch.complex128))
        # site2 should receive 2 (adjacent to site3, hopping coefficient)
        result_adjacent = hamiltonian.apply_within(C_site3, psi_i, C_site2)
        torch.testing.assert_close(result_adjacent, torch.tensor([2.0 + 0j], dtype=torch.complex128))

    def test_hopping_amplitude_in_two_particle_sector(self, hamiltonian: Hamiltonian) -> None:
        """c†₀c₁ connects sites12 (config 6) to sites02 (config 5) with amplitude 2."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        result = hamiltonian.apply_within(C_sites12, psi_i, C_sites02)
        torch.testing.assert_close(result, torch.tensor([2.0 + 0j], dtype=torch.complex128))

    def test_complex_amplitude_preserved(self, hamiltonian: Hamiltonian) -> None:
        """With imaginary input amplitude, the output amplitude is imaginary too."""
        psi_i = torch.tensor([1.0j], dtype=torch.complex128)
        result = hamiltonian.apply_within(C_site0, psi_i, C_site1)
        torch.testing.assert_close(result, torch.tensor([2.0j], dtype=torch.complex128))

    def test_full_1particle_sector_matrix(self, hamiltonian: Hamiltonian) -> None:
        """Apply H to the full 1-particle sector and verify the tridiagonal matrix.

        In the 1-particle sector the Hamiltonian matrix is tridiagonal:
            [[1, 2, 0, 0],
             [2, 1, 2, 0],
             [0, 2, 1, 2],
             [0, 0, 2, 1]]
        where 1 = n coefficient, 2 = t coefficient.
        """
        psi_i = torch.tensor([1.0 + 0j, 1.0 + 0j, 1.0 + 0j, 1.0 + 0j], dtype=torch.complex128)
        result = hamiltonian.apply_within(C_1particle, psi_i, C_1particle)
        # Row sums of the tridiagonal matrix above
        expected = torch.tensor([3.0 + 0j, 5.0 + 0j, 5.0 + 0j, 3.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Tests for find_relative
# ---------------------------------------------------------------------------


class TestFindRelative:
    """find_relative(configs_i, psi_i, count_selected) returns at most count_selected
    new configurations with the highest estimated weights, excluding configs_exclude."""

    def test_finds_both_neighbours_of_site1(self, hamiltonian: Hamiltonian) -> None:
        """From site1 (config 2), connected configs are site0 and site2."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        result = hamiltonian.find_relative(C_site1, psi_i, count_selected=10)
        found = set(result[:, 0].tolist())
        assert found == {1, 4}  # site0 and site2

    def test_count_selected_limits_results(self, hamiltonian: Hamiltonian) -> None:
        """count_selected=1 returns at most 1 config even when 2 neighbours exist."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        result_limited = hamiltonian.find_relative(C_site1, psi_i, count_selected=1)
        result_full = hamiltonian.find_relative(C_site1, psi_i, count_selected=10)
        assert result_limited.size(0) == 1
        assert result_full.size(0) == 2

    def test_no_off_diagonal_from_all_occupied(self, hamiltonian: Hamiltonian) -> None:
        """From |1111⟩, hopping is Pauli-blocked; no off-diagonal connections exist."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        result = hamiltonian.find_relative(C_all, psi_i, count_selected=10)
        assert result.size(0) == 0

    def test_custom_exclude_removes_neighbour(self, hamiltonian: Hamiltonian) -> None:
        """Excluding site2 (config 4) from search, only site0 is returned from site1."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        exclude = torch.cat([C_site1, C_site2])  # exclude self and site2
        result = hamiltonian.find_relative(C_site1, psi_i, count_selected=10, configs_exclude=exclude)
        found = set(result[:, 0].tolist())
        assert found == {1}  # only site0


# ---------------------------------------------------------------------------
# Tests for list_relative
# ---------------------------------------------------------------------------


class TestListRelative:
    """list_relative returns ALL unique new configurations and their accumulated amplitudes,
    unlike find_relative which returns at most count_selected."""

    def test_lists_both_neighbours_of_site1(self, hamiltonian: Hamiltonian) -> None:
        """From site1, list_relative finds BOTH site0 and site2 with amplitude 2 each."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        configs_j, psi_j = hamiltonian.list_relative(C_site1, psi_i)
        assert configs_j.size(0) == 2
        result = {configs_j[i, 0].item(): psi_j[i] for i in range(configs_j.size(0))}
        assert set(result.keys()) == {1, 4}  # site0 and site2
        for amp in result.values():
            torch.testing.assert_close(amp, torch.tensor(2.0 + 0j, dtype=torch.complex128))

    def test_lists_all_connections_from_two_particle_state(self, hamiltonian: Hamiltonian) -> None:
        """From sites12 (config 6), both off-diagonal connections carry amplitude 2."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        configs_j, psi_j = hamiltonian.list_relative(C_sites12, psi_i)
        assert configs_j.size(0) == 2
        result = {configs_j[i, 0].item(): psi_j[i] for i in range(configs_j.size(0))}
        assert set(result.keys()) == {5, 10}  # sites02 and sites13
        for amp in result.values():
            torch.testing.assert_close(amp, torch.tensor(2.0 + 0j, dtype=torch.complex128))

    def test_empty_result_for_fully_occupied(self, hamiltonian: Hamiltonian) -> None:
        """From |1111⟩, hopping is Pauli-blocked; list_relative returns empty tensors."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        configs_j, psi_j = hamiltonian.list_relative(C_all, psi_i)
        assert configs_j.size(0) == 0
        assert psi_j.size(0) == 0

    def test_complex_amplitude_preserved(self, hamiltonian: Hamiltonian) -> None:
        """With imaginary input amplitude, the output amplitudes are imaginary."""
        psi_i = torch.tensor([1.0j], dtype=torch.complex128)
        configs_j, psi_j = hamiltonian.list_relative(C_site1, psi_i)
        assert configs_j.size(0) == 2
        result = {configs_j[i, 0].item(): psi_j[i] for i in range(configs_j.size(0))}
        for amp in result.values():
            torch.testing.assert_close(amp, torch.tensor(2.0j, dtype=torch.complex128))

    def test_amplitudes_accumulate_from_multiple_input_configs(self, hamiltonian: Hamiltonian) -> None:
        """Starting from site0 and site2 (psi=[1,1]), list_relative accumulates contributions:
        - site0 → (c†₁c₀) → site1 with amplitude 2
        - site2 → (c†₁c₂) → site1 with amplitude 2  (accumulated: site1 total = 4)
        - site2 → (c†₃c₂) → site3 with amplitude 2  (site3 total = 2)
        """
        configs_i = torch.cat([C_site0, C_site2])
        psi_i = torch.tensor([1.0 + 0j, 1.0 + 0j], dtype=torch.complex128)
        configs_j, psi_j = hamiltonian.list_relative(configs_i, psi_i)

        assert configs_j.size(0) == 2
        result = {configs_j[i, 0].item(): psi_j[i] for i in range(configs_j.size(0))}
        assert set(result.keys()) == {2, 8}  # site1 and site3
        torch.testing.assert_close(result[2], torch.tensor(4.0 + 0j, dtype=torch.complex128))
        torch.testing.assert_close(result[8], torch.tensor(2.0 + 0j, dtype=torch.complex128))

    def test_custom_exclude_removes_neighbour(self, hamiltonian: Hamiltonian) -> None:
        """Excluding site2 (config 4), list_relative returns only site0 from site1."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        exclude = torch.cat([C_site1, C_site2])  # exclude self and site2
        configs_j, psi_j = hamiltonian.list_relative(C_site1, psi_i, configs_exclude=exclude)
        assert configs_j.size(0) == 1
        assert configs_j[0, 0].item() == 1  # only site0
