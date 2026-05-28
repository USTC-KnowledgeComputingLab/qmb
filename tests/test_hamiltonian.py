import pytest
import torch

from qmp.hamiltonian import Hamiltonian


def _build_two_site_fermion():
    """H = c†₁c₀ + c†₀c₁, 2-site spinless fermion."""
    return Hamiltonian(
        {((1, 1), (0, 0)): -1.0, ((0, 1), (1, 0)): -1.0},
        kind="fermi",
        max_op_number=4,
        devices=["localhost:cpu:0"],
    )


def _build_two_site_boson():
    """Same hopping, bose2 (hard-core boson, no JW sign)."""
    return Hamiltonian(
        {((1, 1), (0, 0)): -1.0, ((0, 1), (1, 0)): -1.0},
        kind="bose2",
        max_op_number=4,
        devices=["localhost:cpu:0"],
    )


def _build_hubbard_2x2():
    """
    4-site spinful Hubbard: 8 spin-orbitals in 1 byte.
    Orbitals: 0=1↑,1=1↓,2=2↑,3=2↓,4=3↑,5=3↓,6=4↑,7=4↓
    2x2 lattice: edges (1,2), (1,3), (2,4), (3,4)
    H = -t sum_{<ij>,s} (c+_{js} c_{is} + h.c.) + U sum_i n_{iu}n_{id}
    """
    t_val = 1.0
    u_val = 4.0
    ham = {}

    def idx(site, spin):
        return site * 2 + spin

    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for i, j in edges:
        for s in (0, 1):
            si = idx(i, s)
            sj = idx(j, s)
            ham[((sj, 1), (si, 0))] = -t_val
            ham[((si, 1), (sj, 0))] = -t_val

    for site in range(4):
        up = idx(site, 0)
        dn = idx(site, 1)
        ham[((up, 1), (up, 0), (dn, 1), (dn, 0))] = u_val

    return Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])


class TestBasicFermion:
    def test_single_hopping_forward(self):
        """H|c†₀|vac⟩ = -|c†₁|vac⟩"""
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert pj.shape == (1,)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))

    def test_hermitian_conjugate(self):
        """H|10⟩ = -|01⟩"""
        h = _build_two_site_fermion()
        ci = torch.tensor([[2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))

    def test_pauli_exclusion(self):
        """|11⟩: both occupied, creating fails → zero."""
        h = _build_two_site_fermion()
        ci = torch.tensor([[3]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[3]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([0.0 + 0.0j], dtype=torch.complex64))

    def test_no_connected_term(self):
        """|1⟩ not connected to |11⟩ by single hopping."""
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[3]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([0.0 + 0.0j], dtype=torch.complex64))

    def test_multiple_contributions(self):
        """Two terms hit same target, amplitudes accumulate."""
        h = Hamiltonian(
            {((1, 1), (0, 0)): -2.0, ((0, 1), (1, 0)): -1.0},
            kind="fermi",
            max_op_number=4,
            devices=["localhost:cpu:0"],
        )
        ci = torch.tensor([[1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1], [2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-0.5 + 0.0j, -2.0 + 0.0j], dtype=torch.complex64))

    def test_complex_coefficients(self):
        """Complex coefficient produces complex output."""
        h = Hamiltonian(
            {((1, 1), (0, 0)): 0.0 + 1.0j},
            kind="fermi",
            max_op_number=4,
            devices=["localhost:cpu:0"],
        )
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([0.0 + 1.0j], dtype=torch.complex64))


class TestBoson:
    def test_no_jw_sign(self):
        h = _build_two_site_boson()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))


class TestDirection:
    def test_forward_matches_backward(self):
        h = _build_two_site_fermion()
        ci = torch.tensor([[1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1], [2]], dtype=torch.uint8)
        fwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="forward")
        bwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="backward")
        assert torch.allclose(fwd, bwd)

    def test_forward_matches_backward_hubbard(self):
        h = _build_hubbard_2x2()
        ci = torch.tensor([[0b00000011]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00001100]], dtype=torch.uint8)
        fwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="forward")
        bwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="backward")
        assert torch.allclose(fwd, bwd)


class TestDeprecation:
    def test_apply_within_warns(self):
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        with pytest.warns(DeprecationWarning, match="apply_within_subspace_in_double_side"):
            result = h.apply_within(ci, pi, cj)
        expected = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(result, expected)


class TestDevices:
    def test_multi_device_raises_on_init(self):
        with pytest.raises(NotImplementedError):
            Hamiltonian(
                {((1, 1), (0, 0)): -1.0},
                kind="fermi",
                max_op_number=4,
                devices=["localhost:cuda:0", "localhost:cuda:1"],
            )


class TestSorted:
    def test_sorted_params_no_crash(self):
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(
            ci, pi, cj, configs_i_sorted=True, configs_j_sorted=True
        )
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))


class TestHubbard:
    def test_diagonal_u_term(self):
        """U term on doubly occupied site gives +U."""
        h = _build_hubbard_2x2()
        ci = torch.tensor([[0b00000011]], dtype=torch.uint8)
        cj = torch.tensor([[0b00000011]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(torch.abs(pj), torch.tensor([4.0], dtype=torch.float32), atol=1e-5)

    def test_hopping_preserves_spin(self):
        """Hopping: H|c†₁↑|vac⟩ has -|c†₂↑|vac⟩ + -|c†₃↑|vac⟩."""
        h = _build_hubbard_2x2()
        ci = torch.tensor([[0b00000001]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00000100], [0b00010000]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j, -1.0 + 0.0j], dtype=torch.complex64))


class TestEdgeCases:
    def test_identity_configs(self):
        """configs_i == configs_j with no diagonal → zero."""
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        pj = h.apply_within_subspace_in_double_side(ci, pi, ci)
        assert torch.allclose(pj, torch.tensor([0.0 + 0.0j], dtype=torch.complex64))

    def test_empty_input_batch(self):
        """Empty configs_i → zero output."""
        h = _build_two_site_fermion()
        ci = torch.empty((0, 1), dtype=torch.uint8)
        pi = torch.empty((0,), dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert pj.shape == (1,)
        assert torch.allclose(pj, torch.tensor([0.0 + 0.0j], dtype=torch.complex64))

    def test_empty_output_batch(self):
        """Empty configs_j → empty output."""
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.empty((0, 1), dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert pj.numel() == 0

    def test_configs_i_superset_of_j(self):
        """Some configs in configs_i not connected to any config_j."""
        h = _build_two_site_fermion()
        ci = torch.tensor([[1], [2], [3]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-0.5 + 0.0j], dtype=torch.complex64))

    def test_configs_j_superset_of_i(self):
        """Some configs in configs_j not reachable → left at zero."""
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0], [1], [2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j], dtype=torch.complex64))

    def test_duplicate_configs_in_i(self):
        """Duplicate configs with different psi → contributions summed."""
        h = _build_two_site_fermion()
        ci = torch.tensor([[1], [1]], dtype=torch.uint8)
        pi = torch.tensor([0.3 + 0.0j, 0.7 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))

    def test_duplicate_configs_in_j(self):
        """Binary search with duplicates hits one occurrence (expected: configs_j should be unique)."""
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2], [2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.isclose(torch.abs(pj).sum(), torch.tensor(1.0, dtype=torch.float32))

    def test_complex_psi_i(self):
        """Complex psi_i multiplies with complex coefficient."""
        h = Hamiltonian(
            {((1, 1), (0, 0)): 1.0 + 0.0j},
            kind="fermi",
            max_op_number=4,
            devices=["localhost:cpu:0"],
        )
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([2.0 + 3.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([2.0 + 3.0j], dtype=torch.complex64))

    def test_larger_batch(self):
        """Exercise sorting with more configs."""
        h = _build_hubbard_2x2()
        ci = torch.tensor(
            [[0b00000001], [0b00000010], [0b00000100], [0b00001000],
             [0b00010000], [0b00100000], [0b01000000], [0b10000000]],
            dtype=torch.uint8,
        )
        pi = torch.ones(8, dtype=torch.complex64)
        cj = torch.tensor(
            [[0b00000001], [0b00000010]],
            dtype=torch.uint8,
        )
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert pj.shape == (2,)

    def test_max_op_number_padding(self):
        """max_op_number=8 for 2-op Hamiltonian → kind=2 padding works."""
        h = Hamiltonian(
            {((1, 1), (0, 0)): -1.0, ((0, 1), (1, 0)): -1.0},
            kind="fermi",
            max_op_number=8,
            devices=["localhost:cpu:0"],
        )
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))

    def test_diagonal_only_hamiltonian(self):
        """Number operators only → only self-config survives."""
        h = Hamiltonian(
            {((0, 1), (0, 0)): 1.0, ((1, 1), (1, 0)): 2.0},
            kind="fermi",
            max_op_number=4,
            devices=["localhost:cpu:0"],
        )
        ci = torch.tensor([[1], [2], [3]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1], [2], [3]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j], dtype=torch.complex64))

    def test_unsorted_input_still_works(self):
        """Unsorted configs_j should still produce correct results."""
        h = _build_two_site_fermion()
        ci = torch.tensor([[1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2], [1], [0]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j, -0.5 + 0.0j, 0.0 + 0.0j], dtype=torch.complex64))


class TestBose2EdgeCases:
    def test_bose2_diagonal(self):
        """Bose2 diagonal number operators."""
        h = Hamiltonian(
            {((0, 1), (0, 0)): 3.0},
            kind="bose2",
            max_op_number=4,
            devices=["localhost:cpu:0"],
        )
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([3.0 + 0.0j], dtype=torch.complex64))

    def test_bose2_multiple_contributions(self):
        """Bose2 with multiple hopping terms, no JW sign flipping."""
        h = Hamiltonian(
            {((1, 1), (0, 0)): -2.0, ((2, 1), (1, 0)): -1.0},
            kind="bose2",
            max_op_number=4,
            devices=["localhost:cpu:0"],
        )
        ci = torch.tensor([[1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-2.0 + 0.0j], dtype=torch.complex64))


class TestInvalidInput:
    def test_invalid_device_string(self):
        with pytest.raises(ValueError, match="Invalid device string"):
            Hamiltonian(
                {((0, 1), (0, 0)): 1.0},
                kind="fermi",
                max_op_number=4,
                devices=["invalid"],
            )

    def test_non_localhost_raises(self):
        with pytest.raises(ValueError, match="Invalid device string"):
            Hamiltonian(
                {((0, 1), (0, 0)): 1.0},
                kind="fermi",
                max_op_number=4,
                devices=["remote:cuda:0"],
            )
