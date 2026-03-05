"""
Tests for qmp.hamiltonian.Hamiltonian.

We use a 2-qubit fermionic Hamiltonian (n_qubytes=1):

    H = c†₀c₁ + c†₁c₀ + n₀ + n₁

where c†ᵢ / cᵢ are creation / annihilation operators and nᵢ = c†ᵢcᵢ is the
number operator at site i.

Operator-key convention in the Hamiltonian dict:
    Key   ((s₀, k₀), (s₁, k₁), ...)  →  operator  O_{k₀,s₀} O_{k₁,s₁}
    with  k=0: annihilation,  k=1: creation.
    The operators are applied right-to-left to the quantum state, i.e. the
    rightmost operator in the expression acts on the state first.

Occupation-number basis for 2 qubits (bit 0 = site 0, bit 1 = site 1):
    |00⟩  →  uint8 value 0
    |10⟩  →  uint8 value 1  (site 0 occupied)
    |01⟩  →  uint8 value 2  (site 1 occupied)
    |11⟩  →  uint8 value 3  (both sites occupied)

In this basis, the Hamiltonian matrix is:
    H = [[0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 2]]
"""

import pytest
import torch

from qmp.hamiltonian import Hamiltonian

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def two_qubit_hamiltonian() -> Hamiltonian:
    """
    2-qubit fermionic Hamiltonian: H = c†₀c₁ + c†₁c₀ + n₀ + n₁.

    Compiled and cached once per test session to avoid repeated JIT overhead.
    """
    return Hamiltonian(
        {
            ((0, 1), (1, 0)): 1.0,  # c†₀c₁
            ((1, 1), (0, 0)): 1.0,  # c†₁c₀
            ((0, 1), (0, 0)): 1.0,  # n₀ = c†₀c₀
            ((1, 1), (1, 0)): 1.0,  # n₁ = c†₁c₁
        },
        kind="fermi",
    )


# ---------------------------------------------------------------------------
# Helpers – configs for 2-qubit system (n_qubytes = 1)
# ---------------------------------------------------------------------------

C_00 = torch.tensor([[0]], dtype=torch.uint8)  # |00⟩
C_10 = torch.tensor([[1]], dtype=torch.uint8)  # |10⟩  (site 0 occupied)
C_01 = torch.tensor([[2]], dtype=torch.uint8)  # |01⟩  (site 1 occupied)
C_11 = torch.tensor([[3]], dtype=torch.uint8)  # |11⟩  (both sites occupied)

ALL_CONFIGS = torch.tensor([[0], [1], [2], [3]], dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Tests for diagonal_term
# ---------------------------------------------------------------------------


class TestDiagonalTerm:
    """diagonal_term(configs) returns the diagonal element H[i,i] for each config."""

    def test_vacuum_diagonal_is_zero(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        result = two_qubit_hamiltonian.diagonal_term(C_00)
        expected = torch.tensor([0.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_single_particle_site0(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        result = two_qubit_hamiltonian.diagonal_term(C_10)
        expected = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_single_particle_site1(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        result = two_qubit_hamiltonian.diagonal_term(C_01)
        expected = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_two_particle_sector(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        result = two_qubit_hamiltonian.diagonal_term(C_11)
        expected = torch.tensor([2.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_all_configs_batch(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """Batched evaluation matches individual results."""
        result = two_qubit_hamiltonian.diagonal_term(ALL_CONFIGS)
        expected = torch.tensor([0.0 + 0j, 1.0 + 0j, 1.0 + 0j, 2.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Tests for apply_within
# ---------------------------------------------------------------------------


class TestApplyWithin:
    """apply_within(configs_i, psi_i, configs_j) computes (H·ψ) restricted to configs_j."""

    def test_apply_to_basis_state_10(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """H|10⟩ = |10⟩ + |01⟩ in the 1-particle sector."""
        configs = torch.cat([C_10, C_01])
        psi_i = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
        result = two_qubit_hamiltonian.apply_within(configs, psi_i, configs)
        expected = torch.tensor([1.0 + 0j, 1.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_apply_to_basis_state_01(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """H|01⟩ = |10⟩ + |01⟩ in the 1-particle sector."""
        configs = torch.cat([C_10, C_01])
        psi_i = torch.tensor([0.0 + 0j, 1.0 + 0j], dtype=torch.complex128)
        result = two_qubit_hamiltonian.apply_within(configs, psi_i, configs)
        expected = torch.tensor([1.0 + 0j, 1.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_apply_to_two_particle_state(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """H|11⟩ = 2|11⟩ (both number operators contribute; hopping is blocked)."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        result = two_qubit_hamiltonian.apply_within(C_11, psi_i, C_11)
        expected = torch.tensor([2.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_no_transition_to_vacuum(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """H maps no configuration to |00⟩."""
        psi_i = torch.ones(4, dtype=torch.complex128)
        result = two_qubit_hamiltonian.apply_within(ALL_CONFIGS, psi_i, C_00)
        expected = torch.tensor([0.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_off_diagonal_10_to_01(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """c†₁c₀|10⟩ → |01⟩ with amplitude 1."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        result = two_qubit_hamiltonian.apply_within(C_10, psi_i, C_01)
        expected = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_off_diagonal_01_to_10(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """c†₀c₁|01⟩ → |10⟩ with amplitude 1."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        result = two_qubit_hamiltonian.apply_within(C_01, psi_i, C_10)
        expected = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_symmetric_state_is_eigenstate(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """(|10⟩+|01⟩)/√2 is an eigenstate of H with eigenvalue 2."""
        configs = torch.cat([C_10, C_01])
        inv_sqrt2 = (0.5**0.5) + 0j
        psi_i = torch.tensor([inv_sqrt2, inv_sqrt2], dtype=torch.complex128)
        result = two_qubit_hamiltonian.apply_within(configs, psi_i, configs)
        torch.testing.assert_close(result, 2 * psi_i)

    def test_antisymmetric_state_has_zero_eigenvalue(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """(|10⟩−|01⟩)/√2 is an eigenstate of H with eigenvalue 0."""
        configs = torch.cat([C_10, C_01])
        inv_sqrt2 = (0.5**0.5) + 0j
        psi_i = torch.tensor([inv_sqrt2, -inv_sqrt2], dtype=torch.complex128)
        result = two_qubit_hamiltonian.apply_within(configs, psi_i, configs)
        expected = torch.zeros(2, dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_complex_amplitude(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """With imaginary input amplitude, the output is also imaginary."""
        psi_i = torch.tensor([1.0j], dtype=torch.complex128)
        result = two_qubit_hamiltonian.apply_within(C_10, psi_i, C_01)
        expected = torch.tensor([1.0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Tests for find_relative
# ---------------------------------------------------------------------------


class TestFindRelative:
    """find_relative returns new configurations connected to the input by H (excluding configs_exclude)."""

    def test_from_10_finds_01(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """From |10⟩ (excluding itself), the only off-diagonal connection is |01⟩."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        result = two_qubit_hamiltonian.find_relative(C_10, psi_i, count_selected=5)
        # result is a uint8 tensor of shape [n_found, n_qubytes]
        assert result.shape[1] == 1
        assert result.size(0) >= 1
        found_values = result[:, 0].tolist()
        assert 2 in found_values  # |01⟩ = uint8 2

    def test_from_01_finds_10(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """From |01⟩ (excluding itself), the only off-diagonal connection is |10⟩."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        result = two_qubit_hamiltonian.find_relative(C_01, psi_i, count_selected=5)
        assert result.shape[1] == 1
        assert result.size(0) >= 1
        found_values = result[:, 0].tolist()
        assert 1 in found_values  # |10⟩ = uint8 1

    def test_from_11_no_off_diagonal(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """From |11⟩ hopping is Pauli-blocked; excluding itself, no connections exist."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        result = two_qubit_hamiltonian.find_relative(C_11, psi_i, count_selected=5)
        assert result.size(0) == 0

    def test_custom_exclude(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """With a custom exclude that contains |01⟩, no off-diagonal result is returned."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        # Exclude both |10⟩ and |01⟩ so that the only off-diagonal partner is gone
        exclude = torch.cat([C_10, C_01])
        result = two_qubit_hamiltonian.find_relative(C_10, psi_i, count_selected=5, configs_exclude=exclude)
        # c†₁c₀|10⟩ → |01⟩ is excluded; no other off-diagonal terms apply
        found_values = result[:, 0].tolist() if result.size(0) > 0 else []
        assert 2 not in found_values


# ---------------------------------------------------------------------------
# Tests for list_relative
# ---------------------------------------------------------------------------


class TestListRelative:
    """list_relative returns (configs_j, psi_j): all new configurations and their accumulated amplitudes."""

    def test_from_10_finds_01_with_amplitude(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """c†₁c₀|10⟩ → |01⟩; excluding |10⟩ (default), |01⟩ is returned with amplitude 1."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        configs_j, psi_j = two_qubit_hamiltonian.list_relative(C_10, psi_i)
        assert configs_j.shape == (1, 1)
        assert torch.equal(configs_j, C_01)
        torch.testing.assert_close(psi_j, torch.tensor([1.0 + 0j], dtype=torch.complex128))

    def test_from_01_finds_10_with_amplitude(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """c†₀c₁|01⟩ → |10⟩; excluding |01⟩ (default), |10⟩ is returned with amplitude 1."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        configs_j, psi_j = two_qubit_hamiltonian.list_relative(C_01, psi_i)
        assert configs_j.shape == (1, 1)
        assert torch.equal(configs_j, C_10)
        torch.testing.assert_close(psi_j, torch.tensor([1.0 + 0j], dtype=torch.complex128))

    def test_from_11_empty_result(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """From |11⟩, hopping is Pauli-blocked; no off-diagonal connections exist."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        configs_j, psi_j = two_qubit_hamiltonian.list_relative(C_11, psi_i)
        assert configs_j.size(0) == 0
        assert psi_j.size(0) == 0

    def test_complex_amplitude_preserved(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """With imaginary input amplitude i, the output amplitude is also i."""
        psi_i = torch.tensor([1.0j], dtype=torch.complex128)
        configs_j, psi_j = two_qubit_hamiltonian.list_relative(C_10, psi_i)
        assert torch.equal(configs_j, C_01)
        torch.testing.assert_close(psi_j, torch.tensor([1.0j], dtype=torch.complex128))

    def test_custom_exclude_reveals_diagonal_term(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """Excluding |01⟩ instead of |10⟩ lets the diagonal term n₀|10⟩=|10⟩ appear."""
        psi_i = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        # exclude |01⟩: the off-diagonal c†₁c₀|10⟩→|01⟩ is excluded, but n₀|10⟩→|10⟩ is not
        configs_j, psi_j = two_qubit_hamiltonian.list_relative(C_10, psi_i, configs_exclude=C_01)
        assert torch.equal(configs_j, C_10)
        torch.testing.assert_close(psi_j, torch.tensor([1.0 + 0j], dtype=torch.complex128))

    def test_accumulated_amplitude_from_multiple_input_configs(self, two_qubit_hamiltonian: Hamiltonian) -> None:
        """
        Starting from both |10⟩ and |01⟩ with amplitudes a and b respectively,
        n₀|10⟩→|10⟩ and c†₀c₁|01⟩→|10⟩ both contribute to the output |10⟩.
        Similarly for |01⟩.  Excluding only |11⟩ so all 1-particle configs can appear.
        """
        configs_i = torch.cat([C_10, C_01])
        psi_i = torch.tensor([2.0 + 0j, 3.0 + 0j], dtype=torch.complex128)
        # Exclude |11⟩ so that both |10⟩ and |01⟩ can appear in the output
        configs_j, psi_j = two_qubit_hamiltonian.list_relative(configs_i, psi_i, configs_exclude=C_11)

        # Build expected: for each output config, sum contributions from all terms
        # |10⟩ receives: n₀|10⟩ → coef=1, psi=2 → +2; c†₀c₁|01⟩ → coef=1, psi=3 → +3; total=5
        # |01⟩ receives: c†₁c₀|10⟩ → coef=1, psi=2 → +2; n₁|01⟩ → coef=1, psi=3 → +3; total=5
        expected_configs = {1: 5.0 + 0j, 2: 5.0 + 0j}

        assert configs_j.size(0) == len(expected_configs)
        for idx in range(configs_j.size(0)):
            config_val = configs_j[idx, 0].item()
            assert config_val in expected_configs
            torch.testing.assert_close(psi_j[idx], torch.tensor(expected_configs[config_val], dtype=torch.complex128))
