#!/usr/bin/env python
"""Simple test comparing PySCF FCI and HAAR FCI solvers."""

from pyscf import gto, scf, fci, ao2mo
from qmp.plugins.pyscf import HAAR

# 1. Create molecule
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = scf.RHF(mol).run()
print(f"HF energy: {mf.e_tot:.8f}")

# 2. PySCF built-in FCI solver
e_pyscf, _ = fci.FCI(mf).kernel()
print(f"PySCF FCI energy: {e_pyscf:.8f}")

# 3. HAAR FCI solver
# IMPORTANT: Must use MO-basis integrals!
# mf.get_hcore() returns AO-basis, need to transform to MO-basis
C = mf.mo_coeff
h1e_mo = C.T @ mf.get_hcore() @ C
eri_mo = ao2mo.restore(1, ao2mo.kernel(mol, C), mol.nao)  # Full 4D MO-basis

solver = HAAR(mol)
print(mol.nao)
solver.sampling_count = 8192
solver.krylov_iteration = 32
solver.local_step = 10
solver.max_cycle = 5
solver.device = "cpu"

e_haar, _ = solver.kernel(h1e_mo, eri_mo, mol.nao, mol.nelec, ecore=mol.energy_nuc())
print(f"HAAR FCI energy: {e_haar:.8f}")
print(f"Error: {abs(e_haar - e_pyscf):.8f}")
