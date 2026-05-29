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


class TestCpuBasicFermion:
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


class TestCpuBoson:
    def test_no_jw_sign(self):
        h = _build_two_site_boson()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))


class TestCpuDirection:
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


class TestCpuDeprecation:
    def test_apply_within_warns(self):
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        with pytest.warns(DeprecationWarning, match="apply_within_subspace_in_double_side"):
            result = h.apply_within(ci, pi, cj)
        expected = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(result, expected)


class TestCpuDevices:
    def test_multi_device_raises_on_init(self):
        with pytest.raises(NotImplementedError):
            Hamiltonian(
                {((1, 1), (0, 0)): -1.0},
                kind="fermi",
                max_op_number=4,
                devices=["localhost:cuda:0", "localhost:cuda:1"],
            )


class TestCpuSorted:
    def test_sorted_params_no_crash(self):
        h = _build_two_site_fermion()
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(
            ci, pi, cj, configs_i_sorted=True, configs_j_sorted=True
        )
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))


class TestCpuHubbard:
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


class TestCpuEdgeCases:
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


class TestCpuBose2EdgeCases:
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


class TestCpuInvalidInput:
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


class TestCpuHeisenberg:
    """Heisenberg spin chain: H = J/2 Σ (S+_i S-_j + h.c.) + Jz Σ S^z_i S^z_j.
    S=1/2 hard-core bosons (bose2), particle_cut=2, each site 0 or 1 boson.
    S+ = c†, S- = c, S^z = n - 1/2.
    S^z_i S^z_j = n_i n_j - n_i/2 - n_j/2 + 1/4."""

    def _build_heisenberg_2site(self, jxy=1.0, jz=1.0):
        ham = {}
        # hopping: Jxy/2 * (S+_0 S-_1 + h.c.)
        ham[((0, 1), (1, 0))] = jxy / 2
        ham[((1, 1), (0, 0))] = jxy / 2
        # n_0 n_1 term
        ham[((0, 1), (0, 0), (1, 1), (1, 0))] = jz
        # -1/2 * n_0
        ham[((0, 1), (0, 0))] = -jz / 2
        # -1/2 * n_1
        ham[((1, 1), (1, 0))] = -jz / 2
        # constant 1/4 omitted
        return Hamiltonian(ham, kind="bose2", max_op_number=4, devices=["localhost:cpu:0"])

    def test_ferromagnetic_ground_state(self):
        """Jxy = Jz = -1 (ferromagnetic): ||11⟩⟩ has energy 0 (both spins aligned)."""
        h = self._build_heisenberg_2site(jxy=-1.0, jz=-1.0)
        # ||11⟩⟩ both occupied: apply H to |11⟩
        ci = torch.tensor([[3]], dtype=torch.uint8)  # |11⟩
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[3]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        # Jxy term: S+_0 S-_1 on |11⟩ → site1=1, subtract→0, site0=0, add→1. parity 0. same config.
        # But wait: S+_0 creates at 0 (already occupied) → fail. S-_1 annihilates at 1 → succeeds.
        # Actually: apply operators right to left: 
        #   S-_1 (kind=0 at site 1): 1→0, parity loop sites 0..0=bit0=1 → parity=1, sign=-1
        #   S+_0 (kind=1 at site 0): 0→1, parity loop sites 0..-1=none → parity=0
        # result: |11⟩ unchanged, sign=-1, contrib = (-jxy/2)*(-1)*1 = jxy/2
        # Same from h.c.: S+_0, S-_1 reversed. h.c. contrib = jxy/2
        # Total hopping contrib = jxy = -1.0
        # Jz S^z S^z: n_0 n_1 contributes +jz = -1.0 since both occupied.
        # -1/2 n_0: -jz/2 * n_0 = 0.5
        # -1/2 n_1: -jz/2 * n_1 = 0.5
        # Total = -1.0 + 0.5 + 0.5 = 0.0 (ferromagnetic ground state, constant term omitted)
        assert torch.allclose(pj, torch.tensor([0.0 + 0.0j], dtype=torch.complex64))

    def test_antiferromagnetic_singlet(self):
        """Jxy = Jz = 1 (antiferromagnetic): check H|01⟩ → Jxy/2 |10⟩ + diagonal terms."""
        h = self._build_heisenberg_2site(jxy=1.0, jz=1.0)
        ci = torch.tensor([[1]], dtype=torch.uint8)  # |01⟩: site0 occupied
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1], [2]], dtype=torch.uint8)  # |01⟩, |10⟩
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        # |01⟩: -n_0/2 = -0.5, n_0 n_1=0 (n_1=0). contrib = -0.5
        # |10⟩: hopping = jxy/2 = 0.5
        assert torch.allclose(pj, torch.tensor([-0.5 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64))

    def test_heisenberg_forward_matches_backward(self):
        h = self._build_heisenberg_2site(jxy=1.0, jz=2.0)
        ci = torch.tensor([[0], [1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0], [1], [2]], dtype=torch.uint8)
        fwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="forward")
        bwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="backward")
        assert torch.allclose(fwd, bwd)


class TestCpuMultiByteConfigs:
    """Systems with n_qubytes > 1 (more than 8 spin-orbitals)."""

    def test_nqubytes_2_hopping(self):
        """10-site spinless fermion chain (n_qubytes=2). byte0=sites 0-7, byte1=sites 8-15.
        H = -t Σ c†_{i} c_{i+1} + h.c."""
        ham = {}
        for i in range(9):
            ham[((i + 1, 1), (i, 0))] = -1.0
            ham[((i, 1), (i + 1, 0))] = -1.0
        h = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[1, 0]], dtype=torch.uint8)  # site 0: byte0 bit0
        cj = torch.tensor([[2, 0]], dtype=torch.uint8)  # site 1: byte0 bit1
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))

    def test_nqubytes_2_cross_byte(self):
        """Hopping across byte boundary: H[c†_8 c_7]|0000000100000000⟩ → -|1000000000000000⟩.
        Site 7 (byte0 bit7=128) → site 8 (byte1 bit0=1). JW sign: parity=0."""
        ham = {((8, 1), (7, 0)): -1.0, ((7, 1), (8, 0)): -1.0}
        h = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[128, 0]], dtype=torch.uint8)  # site 7: byte0 bit7
        cj = torch.tensor([[0, 1]], dtype=torch.uint8)    # site 8: byte1 bit0
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))

    def test_nqubytes_3_forward_backward(self):
        """20-site spinless fermion (n_qubytes=3). Verify forward == backward."""
        ham = {}
        for i in range(19):
            ham[((i + 1, 1), (i, 0))] = -1.0
            ham[((i, 1), (i + 1, 0))] = -1.0
        h = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.uint8)
        fwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="forward")
        bwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="backward")
        assert torch.allclose(fwd, bwd)


class TestCpuMaxOpNumber:
    def test_max_op_number_6(self):
        """max_op_number=6 with 2-op terms: kind=2 padding fills unused slots."""
        h = Hamiltonian(
            {((1, 1), (0, 0)): -1.0, ((0, 1), (1, 0)): -1.0},
            kind="fermi",
            max_op_number=6,
            devices=["localhost:cpu:0"],
        )
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))

    def test_max_op_number_10(self):
        """max_op_number=10 with 2-op terms: larger padding."""
        h = Hamiltonian(
            {((1, 1), (0, 0)): -1.0, ((0, 1), (1, 0)): -1.0},
            kind="fermi",
            max_op_number=10,
            devices=["localhost:cpu:0"],
        )
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        pj = h.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj, torch.tensor([-1.0 + 0.0j], dtype=torch.complex64))


class TestCUDA:
    _fermi_hopping = {((1, 1), (0, 0)): -1.0, ((0, 1), (1, 0)): -1.0}

    @staticmethod
    def _cpu(ham_dict, *, kind="fermi", max_op_number=4):
        return Hamiltonian(ham_dict, kind=kind, max_op_number=max_op_number, devices=["localhost:cpu:0"])

    @staticmethod
    def _cuda(ham_dict, *, kind="fermi", max_op_number=4):
        return Hamiltonian(ham_dict, kind=kind, max_op_number=max_op_number, devices=["localhost:cuda:0"])

    @staticmethod
    def _assert_allclose(a, b, **kwargs):
        assert torch.allclose(a.cpu(), b.cpu(), **kwargs)

    def test_same_as_cpu(self):
        h_cpu = self._cpu(self._fermi_hopping)
        h_cuda = self._cuda(self._fermi_hopping)
        ci = torch.tensor([[1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1], [2]], dtype=torch.uint8)
        self._assert_allclose(
            h_cpu.apply_within_subspace_in_double_side(ci, pi, cj),
            h_cuda.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_forward_backward(self):
        h_cuda = self._cuda(self._fermi_hopping)
        ci = torch.tensor([[1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1], [2]], dtype=torch.uint8)
        fwd = h_cuda.apply_within_subspace_in_double_side(ci, pi, cj, direction="forward")
        bwd = h_cuda.apply_within_subspace_in_double_side(ci, pi, cj, direction="backward")
        self._assert_allclose(fwd, bwd, atol=1e-6)

    def test_boson(self):
        ham = {((1, 1), (0, 0)): -1.0, ((0, 1), (1, 0)): -1.0}
        h_cpu = self._cpu(ham, kind="bose2")
        h_cuda = self._cuda(ham, kind="bose2")
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        self._assert_allclose(
            h_cpu.apply_within_subspace_in_double_side(ci, pi, cj),
            h_cuda.apply_within_subspace_in_double_side(ci, pi, cj),
        )

    def test_complex_coefficient(self):
        ham = {((1, 1), (0, 0)): 0.0 + 1.0j}
        h_cpu = self._cpu(ham)
        h_cuda = self._cuda(ham)
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        self._assert_allclose(
            h_cpu.apply_within_subspace_in_double_side(ci, pi, cj),
            h_cuda.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_pauli_exclusion(self):
        h_cuda = self._cuda(self._fermi_hopping)
        ci = torch.tensor([[3]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[3]], dtype=torch.uint8)
        pj = h_cuda.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(pj.cpu(), torch.tensor([0.0 + 0.0j], dtype=torch.complex64))

    def test_diagonal_only(self):
        ham = {((0, 1), (0, 0)): 1.0, ((1, 1), (1, 0)): 2.0}
        h_cpu = self._cpu(ham)
        h_cuda = self._cuda(ham)
        ci = torch.tensor([[1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1], [2]], dtype=torch.uint8)
        self._assert_allclose(
            h_cpu.apply_within_subspace_in_double_side(ci, pi, cj),
            h_cuda.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_empty_batches(self):
        h_cuda = self._cuda(self._fermi_hopping)
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.empty((0, 1), dtype=torch.uint8)
        pj = h_cuda.apply_within_subspace_in_double_side(ci, pi, cj)
        assert pj.numel() == 0

    def test_unsorted_configs(self):
        h_cuda = self._cuda(self._fermi_hopping)
        ci = torch.tensor([[1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2], [1], [0]], dtype=torch.uint8)
        pj = h_cuda.apply_within_subspace_in_double_side(ci, pi, cj)
        expected = torch.tensor([-1.0 + 0.0j, -0.5 + 0.0j, 0.0 + 0.0j], dtype=torch.complex64)
        assert torch.allclose(pj.cpu(), expected)

    def test_superset_configs_j(self):
        h_cuda = self._cuda(self._fermi_hopping)
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0], [1], [2]], dtype=torch.uint8)
        pj = h_cuda.apply_within_subspace_in_double_side(ci, pi, cj)
        expected = torch.tensor([0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j], dtype=torch.complex64)
        assert torch.allclose(pj.cpu(), expected)

    def test_hubbard_hopping(self):
        ham = {
            ((2, 1), (0, 0)): -1.0, ((0, 1), (2, 0)): -1.0,
            ((3, 1), (1, 0)): -1.0, ((1, 1), (3, 0)): -1.0,
        }
        h_cpu = self._cpu(ham)
        h_cuda = self._cuda(ham)
        ci = torch.tensor([[0b00000001]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00000100]], dtype=torch.uint8)
        self._assert_allclose(
            h_cpu.apply_within_subspace_in_double_side(ci, pi, cj),
            h_cuda.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_hubbard_u_term(self):
        ham = {((0, 1), (0, 0), (1, 1), (1, 0)): 4.0}
        h_cpu = self._cpu(ham)
        h_cuda = self._cuda(ham)
        ci = torch.tensor([[0b00000011]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00000011]], dtype=torch.uint8)
        self._assert_allclose(
            h_cpu.apply_within_subspace_in_double_side(ci, pi, cj),
            h_cuda.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )


