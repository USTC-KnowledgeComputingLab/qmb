import pytest
import torch

from qmp.hamiltonian import Hamiltonian, SpinSeparatedHamiltonian


class TestSpinSeparated:
    def _build_hubbard_2x2_ham(self):
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
        return ham

    def test_matches_regular_hamiltonian(self):
        ham = self._build_hubbard_2x2_ham()
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=4, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[0b00000011]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00001100]], dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_hubbard_full_identity(self):
        ham = self._build_hubbard_2x2_ham()
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=4, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[0b00000011], [0b00001100], [0b00001111]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00000011], [0b00001100], [0b00001111]], dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_hubbard_hopping_cross_spin(self):
        ham = self._build_hubbard_2x2_ham()
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=4, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[0b00000011]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00000110]], dtype=torch.uint8)
        assert torch.allclose(
            torch.abs(h_ref.apply_within_subspace_in_double_side(ci, pi, cj)),
            torch.abs(h_spin.apply_within_subspace_in_double_side(ci, pi, cj)),
            atol=1e-6,
        )

    def test_forward_backward(self):
        ham = self._build_hubbard_2x2_ham()
        h = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=4, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[0b00000011], [0b00001100]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00000011], [0b00001100]], dtype=torch.uint8)
        fwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="forward")
        bwd = h.apply_within_subspace_in_double_side(ci, pi, cj, direction="backward")
        assert torch.allclose(fwd, bwd, atol=1e-6)

    def test_two_site_fermion(self):
        ham = {((1, 1), (0, 0)): -1.0, ((0, 1), (1, 0)): -1.0}
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=1, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2]], dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_hubbard_multi_config(self):
        ham = self._build_hubbard_2x2_ham()
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=4, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[0b00000011], [0b00001100], [0b00000001]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00000011], [0b00001100], [0b00000100]], dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_hubbard_identity(self):
        ham = self._build_hubbard_2x2_ham()
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=4, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[0b00000011], [0b00001100]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00000011], [0b00001100]], dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_hubbard_superset_configs(self):
        ham = self._build_hubbard_2x2_ham()
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=4, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[0b00000011]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00000000], [0b00000011], [0b00001100]], dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_hubbard_empty_output(self):
        ham = self._build_hubbard_2x2_ham()
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=4, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[0b00000011]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.empty((0, 1), dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_hubbard_complex_coefficient(self):
        ham = {((0, 1), (2, 0)): 0.0 + 1.0j, ((2, 1), (0, 0)): 0.0 - 1.0j}
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=3, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[4]], dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_different_n_up(self):
        ham = {((0, 1), (2, 0)): -1.0, ((2, 1), (0, 0)): -1.0}
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=2, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[1]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[4]], dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_nqubytes_2_chain(self):
        ham = {}
        for i in range(9):
            ham[((i + 1, 1), (i, 0))] = -1.0
            ham[((i, 1), (i + 1, 0))] = -1.0
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=5, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[1, 0]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2, 0]], dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_nqubytes_3_chain(self):
        ham = {}
        for i in range(17):
            ham[((i + 1, 1), (i, 0))] = -1.0
            ham[((i, 1), (i + 1, 0))] = -1.0
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=9, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[1, 0, 0]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[2, 0, 0]], dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_nqubytes_2_cross_byte_hop(self):
        ham = {}
        for i in range(9):
            ham[((i + 1, 1), (i, 0))] = -1.0
            ham[((i, 1), (i + 1, 0))] = -1.0
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=5, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[128, 0]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0, 1]], dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )

    def test_nqubytes_3_cross_byte_hop(self):
        ham = {}
        for i in range(17):
            ham[((i + 1, 1), (i, 0))] = -1.0
            ham[((i, 1), (i + 1, 0))] = -1.0
        h_ref = Hamiltonian(ham, kind="fermi", max_op_number=4, devices=["localhost:cpu:0"])
        h_spin = SpinSeparatedHamiltonian(ham, kind="fermi", n_up=9, max_op_number=4, devices=["localhost:cpu:0"])
        ci = torch.tensor([[0, 64, 0]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0, 0, 1]], dtype=torch.uint8)
        assert torch.allclose(
            h_ref.apply_within_subspace_in_double_side(ci, pi, cj),
            h_spin.apply_within_subspace_in_double_side(ci, pi, cj),
            atol=1e-6,
        )


class TestSpinSeparatedCUDA:
    @staticmethod
    def _cpu(ham_dict, *, n_up, kind="fermi", max_op_number=4):
        return SpinSeparatedHamiltonian(ham_dict, kind=kind, n_up=n_up, max_op_number=max_op_number, devices=["localhost:cpu:0"])

    @staticmethod
    def _cuda(ham_dict, *, n_up, kind="fermi", max_op_number=4):
        return SpinSeparatedHamiltonian(ham_dict, kind=kind, n_up=n_up, max_op_number=max_op_number, devices=["localhost:cuda:0"])

    def test_same_as_cpu(self):
        ham = {((1, 1), (0, 0)): -1.0, ((0, 1), (1, 0)): -1.0}
        h_cpu = self._cpu(ham, n_up=1)
        h_cuda = self._cuda(ham, n_up=1)
        ci = torch.tensor([[1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1], [2]], dtype=torch.uint8)
        cpu_result = h_cpu.apply_within_subspace_in_double_side(ci, pi, cj)
        cuda_result = h_cuda.apply_within_subspace_in_double_side(ci, pi, cj)
        assert torch.allclose(cpu_result.cpu(), cuda_result.cpu(), atol=1e-6)

    def test_hubbard_cuda(self):
        t_val, u_val = 1.0, 4.0
        ham = {}
        def idx(site, spin):
            return site * 2 + spin
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        for i, j in edges:
            for s in (0, 1):
                si, sj = idx(i, s), idx(j, s)
                ham[((sj, 1), (si, 0))] = -t_val
                ham[((si, 1), (sj, 0))] = -t_val
        for site in range(4):
            up, dn = idx(site, 0), idx(site, 1)
            ham[((up, 1), (up, 0), (dn, 1), (dn, 0))] = u_val
        h_cpu = self._cpu(ham, n_up=4)
        h_cuda = self._cuda(ham, n_up=4)
        ci = torch.tensor([[0b00000011], [0b00001100]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[0b00000011], [0b00001100]], dtype=torch.uint8)
        assert torch.allclose(
            h_cpu.apply_within_subspace_in_double_side(ci, pi, cj).cpu(),
            h_cuda.apply_within_subspace_in_double_side(ci, pi, cj).cpu(),
            atol=1e-6,
        )

    def test_forward_backward_cuda(self):
        ham = {((1, 1), (0, 0)): -1.0, ((0, 1), (1, 0)): -1.0}
        h_cuda = self._cuda(ham, n_up=1)
        ci = torch.tensor([[1], [2]], dtype=torch.uint8)
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        cj = torch.tensor([[1], [2]], dtype=torch.uint8)
        fwd = h_cuda.apply_within_subspace_in_double_side(ci, pi, cj, direction="forward")
        bwd = h_cuda.apply_within_subspace_in_double_side(ci, pi, cj, direction="backward")
        assert torch.allclose(fwd.cpu(), bwd.cpu(), atol=1e-6)
