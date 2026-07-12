"""MLP autoregressive neural quantum states (arXiv:2109.12606).

Each site owns an independent MLP predicting the conditional log-amplitude of the
next site given all previous sites, plus a shared phase network. Three variants
differ only in their particle-number conservation constraints:

- :class:`WaveFunctionNormal` — no conservation, arbitrary ``physical_dim``.
- :class:`WaveFunctionElectron` — total electron number conserved.
- :class:`WaveFunctionElectronUpDown` — spin-up and spin-down numbers conserved.

Configurations are bit-packed uint8 arrays ``[batch_size, n_qubytes]``. Sites are
stored internally as integer state indices ``[batch_size, sites]`` in the range
``[0, states_per_site)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

from ..utility.bitspack import pack_int, unpack_int
from ._autoregressive import (
    apply_mask,
    gumbel_topk_step,
    mask_electron,
    mask_electron_up_down,
    normalize_log_amplitude,
    sample_step,
)

if TYPE_CHECKING:
    from jax import Array


def _bit_size_for(physical_dim: int) -> int:
    """Smallest packing bit width in ``{1, 2, 4, 8}`` holding ``physical_dim`` states."""
    if physical_dim <= 2:
        return 1
    if physical_dim <= 4:
        return 2
    if physical_dim <= 16:
        return 4
    if physical_dim <= 256:
        return 8
    raise ValueError(f"physical_dim must be <= 256, got {physical_dim}.")


def _resolve_ordering(ordering: int | list[int], sites: int) -> tuple[int, ...]:
    """Resolve the ordering specification to an explicit permutation tuple."""
    if isinstance(ordering, int):
        if ordering == 1:
            return tuple(range(sites))
        if ordering == -1:
            return tuple(reversed(range(sites)))
        raise ValueError(f"integer ordering must be +1 or -1, got {ordering}.")
    if sorted(ordering) != list(range(sites)):
        raise ValueError("ordering list must be a permutation of range(sites).")
    return tuple(ordering)


def _invert_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    """Return the inverse of a permutation tuple."""
    inverse = [0] * len(permutation)
    for position, value in enumerate(permutation):
        inverse[value] = position
    return tuple(inverse)


class _Linear(nnx.Module):
    """Dense layer that also supports a zero-width input (bias only).

    When ``zero_init`` is set, the kernel and bias start at zero so the layer
    initially outputs a constant. Used for output heads so the initial
    conditional distribution is near-uniform (maximum entropy).
    """

    def __init__(self, in_features: int, out_features: int, *, zero_init: bool = False, rngs: nnx.Rngs) -> None:
        self.in_features = in_features
        self.out_features = out_features
        if in_features == 0:
            self.bias = nnx.Param(jnp.zeros((out_features,), dtype=jnp.float64))
            self.linear = None
        else:
            self.bias = None
            kernel_init = nnx.initializers.zeros if zero_init else nnx.initializers.lecun_normal()
            self.linear = nnx.Linear(
                in_features, out_features, kernel_init=kernel_init, param_dtype=jnp.float64, rngs=rngs
            )

    def __call__(self, inputs: Array) -> Array:
        if self.linear is None:
            batch_size = inputs.shape[0]
            bias = self.bias
            assert bias is not None
            return jnp.broadcast_to(bias[...], (batch_size, self.out_features))
        return self.linear(inputs)


class _MLP(nnx.Module):
    """Multi-layer perceptron with SiLU activations between linear layers.

    When ``zero_output`` is set the final layer is zero-initialised so the whole
    MLP initially outputs zeros.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: tuple[int, ...],
        *,
        zero_output: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        dimensions = [in_features, *hidden_size, out_features]
        last_index = len(dimensions) - 2
        self.layers = nnx.List(
            [
                _Linear(
                    dimensions[index],
                    dimensions[index + 1],
                    zero_init=zero_output and index == last_index,
                    rngs=rngs,
                )
                for index in range(len(dimensions) - 1)
            ]
        )

    def __call__(self, inputs: Array) -> Array:
        activation = inputs
        last_index = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            activation = layer(activation)
            if index != last_index:
                activation = jax.nn.silu(activation)
        return activation


class _MLPWaveFunctionBase(nnx.Module):
    """Shared autoregressive machinery for the three MLP variants.

    Subclasses configure ``sites``, ``states_per_site``, ``feature_per_site``,
    the packing layout, and implement :meth:`_local_mask`.
    """

    sites: int
    states_per_site: int
    feature_per_site: int
    _ordering: tuple[int, ...]
    _ordering_reversed: tuple[int, ...]

    def _build_networks(self, hidden_size: tuple[int, ...], rngs: nnx.Rngs) -> None:
        self.amplitude = nnx.List(
            [
                _MLP(site_index * self.feature_per_site, self.states_per_site, hidden_size, zero_output=True, rngs=rngs)
                for site_index in range(self.sites)
            ]
        )
        self.phase = _MLP(self.sites * self.feature_per_site, 1, hidden_size, zero_output=True, rngs=rngs)

    # ---- variant hooks (overridden by subclasses) ----

    def _config_to_site_values(self, configs: Array) -> Array:
        """Decode bit-packed configs into internal-order site indices ``[batch, sites]``."""
        raise NotImplementedError

    def _site_values_to_config(self, site_values: Array) -> Array:
        """Encode internal-order site indices back into bit-packed configs."""
        raise NotImplementedError

    def _site_values_to_features(self, site_values: Array) -> Array:
        """Convert site indices ``[batch, width]`` to MLP input features ``[batch, width * feature_per_site]``."""
        raise NotImplementedError

    def _local_mask(self, prefix_values: Array, site_index: int) -> Array:
        """Boolean mask ``[batch, states_per_site]`` of allowed next states."""
        raise NotImplementedError

    # ---- shared helpers ----

    def _ordering_array(self) -> Array:
        return jnp.asarray(self._ordering, dtype=jnp.int32)

    def _ordering_reversed_array(self) -> Array:
        return jnp.asarray(self._ordering_reversed, dtype=jnp.int32)

    def _conditional_log_amplitude(self, prefix_values: Array, site_index: int) -> Array:
        """Normalised conditional log-amplitude ``[batch, states_per_site]`` at ``site_index``."""
        features = self._site_values_to_features(prefix_values)
        raw = self.amplitude[site_index](features)
        masked = apply_mask(raw, self._local_mask(prefix_values, site_index))
        return normalize_log_amplitude(masked, axis=-1)

    def __call__(self, configs: Array) -> Array:
        site_values = self._config_to_site_values(configs)
        batch_size = site_values.shape[0]
        batch_indices = jnp.arange(batch_size)

        total_log_amplitude = jnp.zeros((batch_size,), dtype=jnp.float64)
        for site_index in range(self.sites):
            conditional = self._conditional_log_amplitude(site_values[:, :site_index], site_index)
            selected = conditional[batch_indices, site_values[:, site_index]]
            total_log_amplitude = total_log_amplitude + selected

        features = self._site_values_to_features(site_values)
        total_phase = self.phase(features)[:, 0]
        return jnp.exp(total_log_amplitude + 1j * total_phase)

    def generate(self, batch_size: int, *, key: Array) -> tuple[Array, Array, Array]:
        site_values = jnp.zeros((batch_size, 0), dtype=jnp.int32)
        for site_index in range(self.sites):
            conditional = self._conditional_log_amplitude(site_values, site_index)
            key, subkey = jax.random.split(key)
            choice = sample_step(conditional, subkey).astype(jnp.int32)
            site_values = jnp.concatenate([site_values, choice[:, None]], axis=1)

        configs = self._site_values_to_config(site_values)
        unique_configs, counts = jnp.unique(configs, axis=0, return_counts=True)
        psi = self(unique_configs)
        return unique_configs, psi, counts

    def generate_unique(self, batch_size: int, *, key: Array) -> tuple[Array, Array]:
        beam_width = batch_size
        beam_values = jnp.zeros((beam_width, 0), dtype=jnp.int32)
        beam_log_prob = jnp.full((beam_width,), 0.0, dtype=jnp.float64)
        beam_perturbed = jnp.full((beam_width,), 0.0, dtype=jnp.float64)
        beam_valid = jnp.arange(beam_width) == 0

        for site_index in range(self.sites):
            conditional = self._conditional_log_amplitude(beam_values, site_index)
            child_conditional_log_prob = 2.0 * conditional
            key, subkey = jax.random.split(key)
            child_log_prob, child_perturbed, child_valid = gumbel_topk_step(
                beam_log_prob, beam_perturbed, beam_valid, child_conditional_log_prob, subkey
            )

            states = self.states_per_site
            tiled_prefix = jnp.broadcast_to(beam_values[:, None, :], (beam_width, states, site_index)).reshape(
                beam_width * states, site_index
            )
            next_states = jnp.broadcast_to(jnp.arange(states, dtype=jnp.int32)[None, :], (beam_width, states)).reshape(
                beam_width * states, 1
            )
            candidates = jnp.concatenate([tiled_prefix, next_states], axis=1)

            flat_perturbed = child_perturbed.reshape(-1)
            selected = jnp.argsort(flat_perturbed)[::-1][:beam_width]

            beam_values = candidates[selected]
            beam_log_prob = child_log_prob.reshape(-1)[selected]
            beam_perturbed = flat_perturbed[selected]
            beam_valid = child_valid.reshape(-1)[selected]

        configs = self._site_values_to_config(beam_values)
        valid_mask = jnp.asarray(beam_valid)
        configs = configs[valid_mask]
        psi = self(configs)
        return configs, psi


class WaveFunctionNormal(_MLPWaveFunctionBase):
    """MLP wave function without any particle-number conservation."""

    def __init__(
        self,
        *,
        sites: int,
        physical_dim: int,
        hidden_size: tuple[int, ...],
        ordering: int | list[int] = 1,
        rngs: nnx.Rngs,
    ) -> None:
        self.sites = sites
        self.physical_dim = physical_dim
        self.states_per_site = physical_dim
        self.feature_per_site = 1
        self._bit_size = _bit_size_for(physical_dim)
        self._ordering = _resolve_ordering(ordering, sites)
        self._ordering_reversed = _invert_permutation(self._ordering)
        self._build_networks(hidden_size, rngs)

    def _config_to_site_values(self, configs: Array) -> Array:
        site_values = unpack_int(configs, size=self._bit_size, last_dim=self.sites).astype(jnp.int32)
        return site_values[:, self._ordering_reversed_array()]

    def _site_values_to_config(self, site_values: Array) -> Array:
        user_order = site_values[:, self._ordering_array()]
        return pack_int(user_order.astype(jnp.uint8), size=self._bit_size)

    def _site_values_to_features(self, site_values: Array) -> Array:
        return site_values.astype(jnp.float64)

    def _local_mask(self, prefix_values: Array, site_index: int) -> Array:
        batch_size = prefix_values.shape[0]
        return jnp.ones((batch_size, self.states_per_site), dtype=bool)


class WaveFunctionElectron(_MLPWaveFunctionBase):
    """MLP wave function conserving the total electron number."""

    def __init__(
        self,
        *,
        sites: int,
        electrons: int,
        hidden_size: tuple[int, ...],
        ordering: int | list[int] = 1,
        rngs: nnx.Rngs,
    ) -> None:
        self.sites = sites
        self.electrons = electrons
        self.states_per_site = 2
        self.feature_per_site = 1
        self._bit_size = 1
        self._ordering = _resolve_ordering(ordering, sites)
        self._ordering_reversed = _invert_permutation(self._ordering)
        self._build_networks(hidden_size, rngs)

    def _config_to_site_values(self, configs: Array) -> Array:
        site_values = unpack_int(configs, size=1, last_dim=self.sites).astype(jnp.int32)
        return site_values[:, self._ordering_reversed_array()]

    def _site_values_to_config(self, site_values: Array) -> Array:
        user_order = site_values[:, self._ordering_array()]
        return pack_int(user_order.astype(jnp.uint8), size=1)

    def _site_values_to_features(self, site_values: Array) -> Array:
        return site_values.astype(jnp.float64)

    def _local_mask(self, prefix_values: Array, site_index: int) -> Array:
        electron_count = jnp.sum(prefix_values, axis=1)
        sites_filled = jnp.asarray(site_index)
        return mask_electron(electron_count, sites_filled, self.sites, self.electrons)


class WaveFunctionElectronUpDown(_MLPWaveFunctionBase):
    """MLP wave function conserving spin-up and spin-down electron numbers.

    Each site is a pair of qubits (spin-up, spin-down) encoded as the state index
    ``up * 2 + down`` in ``[0, 4)``.
    """

    def __init__(
        self,
        *,
        double_sites: int,
        spin_up: int,
        spin_down: int,
        hidden_size: tuple[int, ...],
        ordering: int | list[int] = 1,
        rngs: nnx.Rngs,
    ) -> None:
        if double_sites % 2 != 0:
            raise ValueError(f"double_sites must be even, got {double_sites}.")
        self.double_sites = double_sites
        self.sites = double_sites // 2
        self.spin_up = spin_up
        self.spin_down = spin_down
        self.states_per_site = 4
        self.feature_per_site = 2
        self._ordering = _resolve_ordering(ordering, self.sites)
        self._ordering_reversed = _invert_permutation(self._ordering)
        self._build_networks(hidden_size, rngs)

    def _config_to_site_values(self, configs: Array) -> Array:
        qubits = unpack_int(configs, size=1, last_dim=self.double_sites).astype(jnp.int32)
        pairs = qubits.reshape(qubits.shape[0], self.sites, 2)
        site_values = pairs[:, :, 0] * 2 + pairs[:, :, 1]
        return site_values[:, self._ordering_reversed_array()]

    def _site_values_to_config(self, site_values: Array) -> Array:
        user_order = site_values[:, self._ordering_array()]
        up = user_order // 2
        down = user_order % 2
        qubits = jnp.stack([up, down], axis=-1).reshape(user_order.shape[0], self.double_sites)
        return pack_int(qubits.astype(jnp.uint8), size=1)

    def _site_values_to_features(self, site_values: Array) -> Array:
        up = site_values // 2
        down = site_values % 2
        interleaved = jnp.stack([up, down], axis=-1).reshape(site_values.shape[0], site_values.shape[1] * 2)
        return interleaved.astype(jnp.float64)

    def _local_mask(self, prefix_values: Array, site_index: int) -> Array:
        up_count = jnp.sum(prefix_values // 2, axis=1)
        down_count = jnp.sum(prefix_values % 2, axis=1)
        sites_filled = jnp.asarray(site_index)
        mask = mask_electron_up_down(up_count, down_count, sites_filled, self.sites, self.spin_up, self.spin_down)
        return mask.reshape(mask.shape[0], 4)
