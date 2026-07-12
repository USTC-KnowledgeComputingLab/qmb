"""Compute the energy of a chopped wavefunction via apply_within_subspace.

Usage:
    uv run --with torch python test.py <chopped.pth> <fcidump.gz>

Loads a chopped wavefunction (config + psi) saved by the old torch pipeline,
builds the Hamiltonian from a FCIDUMP file using the JAX fcidump model, applies
H within the config subspace, and reports the variational energy

    E = <psi|H|psi> / <psi|psi>.
"""

from __future__ import annotations

import argparse
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import torch

from qmp.models.fcidump import Model, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute energy via apply_within_subspace.")
    parser.add_argument("chopped", help="Path to the chopped .pth file (keys: config, psi).")
    parser.add_argument("fcidump", help="Path to the FCIDUMP (.gz) file.")
    args = parser.parse_args()

    print(f"Loading wavefunction from {args.chopped}", file=sys.stderr)
    data = torch.load(args.chopped, map_location="cpu", weights_only=True)
    config_torch = data["config"]  # [N, n_qubytes] uint8
    psi_torch = data["psi"]  # [N] complex128
    batch_size, n_qubytes = config_torch.shape
    print(f"  config: {tuple(config_torch.shape)} {config_torch.dtype}", file=sys.stderr)
    print(f"  psi:    {tuple(psi_torch.shape)} {psi_torch.dtype}", file=sys.stderr)

    configs = jnp.asarray(config_torch.numpy(), dtype=jnp.uint8)
    psi_complex = psi_torch.numpy()
    psi_i = jnp.stack(
        [jnp.asarray(psi_complex.real, dtype=jnp.float64), jnp.asarray(psi_complex.imag, dtype=jnp.float64)],
        axis=1,
    )  # [N, 2] real/imag

    print(f"Building Hamiltonian from {args.fcidump}", file=sys.stderr)
    model = Model(ModelConfig(model_path=args.fcidump))
    if model.n_qubits != n_qubytes * 8:
        print(
            f"WARNING: model n_qubits={model.n_qubits} but config carries {n_qubytes * 8} qubit slots",
            file=sys.stderr,
        )

    print(f"Applying H within subspace over {batch_size} configs", file=sys.stderr)
    h_psi = model.apply_within_subspace(configs, psi_i, configs)  # [N, 2]

    h_psi_complex = h_psi[:, 0] + 1j * h_psi[:, 1]
    psi_i_complex = psi_i[:, 0] + 1j * psi_i[:, 1]

    numerator = jnp.sum(jnp.conjugate(psi_i_complex) * h_psi_complex)
    denominator = jnp.sum(jnp.abs(psi_i_complex) ** 2)
    energy = (numerator / denominator).real

    print(f"<psi|psi>       = {float(denominator):.10f}")
    print(f"<psi|H|psi>     = {float(numerator.real):.10f} (imag {float(numerator.imag):.3e})")
    print(f"energy          = {float(energy):.10f}")


if __name__ == "__main__":
    main()
