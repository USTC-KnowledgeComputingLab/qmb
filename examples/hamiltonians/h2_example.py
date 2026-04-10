from pathlib import Path
from qmp.plugins.hamiltonians import WaveFunction, Hamiltonian


data_dir = Path(__file__).parent
hamiltonian = Hamiltonian(data_dir / "H2.FCIDUMP")
wave = WaveFunction().load(data_dir / "H2.wavefunction")
assert wave.config is not None and wave.psi is not None

print("Loaded wavefunction:")
print(f"  orbit_number: {wave.orbit_number}")
print(f"  config shape: {wave.config.shape}")
print(f"  psi: {wave.psi}")

diag = hamiltonian.diagonal_term(wave)
print(f"\nDiagonal terms: {diag.psi}")

rel = hamiltonian.list_relative(wave)
rel.dump(data_dir / "rel.wavefunction")
print("\nList relative (configs connected by Hamiltonian):")
print(f"  config : {rel.config}")

found = hamiltonian.find_relative(wave, count_selected=10)
found.dump(data_dir / "found.wavefunction")
print("\nFind relative:")
print(f"  config : {found.config}")

result = hamiltonian.apply_within(wave.to("cuda"), found.to("cuda"))
result.dump(data_dir / "result.wavefunction")
print("\nApply within (H|wave⟩ onto found configs):")
print(f"  result psi: {result.psi}")
