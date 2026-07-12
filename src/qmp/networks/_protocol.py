"""Network interface protocol for variational wave-function ansatzes.

All networks (MLP, Transformers, ...) implement :class:`NetworkProto`. Networks
represent a neural quantum state |psi>: they evaluate amplitudes for given
configurations and sample configurations from the Born distribution |psi|^2.

Configurations are bit-packed uint8 arrays of shape ``[batch_size, n_qubytes]``,
matching the format consumed by the Hamiltonian subsystem. Amplitudes are
complex128 arrays of shape ``[batch_size]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jax import Array


@runtime_checkable
class NetworkProto(Protocol):
    """Interface implemented by all variational wave-function networks."""

    def __call__(self, configs: Array) -> Array:
        """Evaluate wave-function amplitudes for the given configurations.

        Parameters
        ----------
        configs : Array
            Bit-packed uint8 configurations of shape ``[batch_size, n_qubytes]``.

        Returns
        -------
        Array
            Complex128 amplitudes of shape ``[batch_size]``.
        """
        ...

    def generate(self, batch_size: int, *, key: Array) -> tuple[Array, Array, Array]:
        """Sample configurations from |psi|^2 with replacement.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw before deduplication.
        key : Array
            A JAX PRNG key.

        Returns
        -------
        tuple[Array, Array, Array]
            ``(configs, psi, counts)`` where ``configs`` are the unique
            bit-packed configurations, ``psi`` their complex amplitudes, and
            ``counts`` the number of times each was sampled.
        """
        ...

    def generate_unique(self, batch_size: int, *, key: Array) -> tuple[Array, Array]:
        """Sample unique configurations without replacement (Gumbel top-K).

        Parameters
        ----------
        batch_size : int
            Maximum number of unique configurations to return (beam width).
        key : Array
            A JAX PRNG key.

        Returns
        -------
        tuple[Array, Array]
            ``(configs, psi)`` where ``configs`` are the unique bit-packed
            configurations (at most ``batch_size`` of them) and ``psi`` their
            complex amplitudes.
        """
        ...
