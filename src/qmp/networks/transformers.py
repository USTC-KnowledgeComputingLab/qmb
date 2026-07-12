"""Transformer autoregressive neural quantum states (dense decoder).

A single causal transformer decoder predicts every site's conditional
log-amplitude and phase. Three variants differ only in their particle-number
conservation constraints, mirroring the MLP variants:

- :class:`WaveFunctionNormal` — no conservation, arbitrary ``physical_dim``.
- :class:`WaveFunctionElectron` — total electron number conserved.
- :class:`WaveFunctionElectronUpDown` — spin-up and spin-down numbers conserved.

Amplitude evaluation (:meth:`__call__`) runs the decoder once over the whole
sequence with a causal mask. Sampling (:meth:`generate` / :meth:`generate_unique`)
uses incremental key/value caching: at each site only the newly appended token is
fed through the decoder, and the cached keys/values are reordered along the batch
axis to follow the beam search. The decoder uses dense feed-forward blocks (no
mixture-of-experts).

Output heads are zero-initialised so the initial conditional distribution is
near-uniform (maximum entropy), which is a neutral starting point for VMC.
"""

from __future__ import annotations

import functools
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


class _PositionalEmbedding(nnx.Module):
    """Per-position, per-token embedding table of shape ``[sites, physical_dim, embedding_dim]``.

    ``position`` selects which site rows are read, so a single incremental token
    can be embedded at its true position.
    """

    def __init__(self, sites: int, physical_dim: int, embedding_dim: int, *, rngs: nnx.Rngs) -> None:
        initializer = nnx.initializers.normal(stddev=0.02)
        self.table = nnx.Param(initializer(rngs.params(), (sites, physical_dim, embedding_dim), jnp.float64))

    @functools.partial(nnx.jit, static_argnums=(2,))
    def __call__(self, tokens: Array, position: int = 0) -> Array:
        sequence_length = tokens.shape[1]
        table = self.table[...][position : position + sequence_length]
        # table: [seq, physical_dim, emb]; tokens: [batch, seq] -> [batch, seq, emb]
        return jnp.take_along_axis(table[None], tokens[:, :, None, None], axis=2)[:, :, 0, :]


class _FeedForward(nnx.Module):
    """Dense feed-forward block: Linear -> GELU -> Linear."""

    def __init__(self, embedding_dim: int, hidden_dim: int, *, rngs: nnx.Rngs) -> None:
        self.up = nnx.Linear(embedding_dim, hidden_dim, param_dtype=jnp.float64, rngs=rngs)
        self.down = nnx.Linear(hidden_dim, embedding_dim, param_dtype=jnp.float64, rngs=rngs)

    @nnx.jit
    def __call__(self, inputs: Array) -> Array:
        return self.down(jax.nn.gelu(self.up(inputs)))


class _DecoderUnit(nnx.Module):
    """Pre-norm transformer decoder block with causal self-attention and dense FFN."""

    def __init__(self, embedding_dim: int, heads_num: int, feed_forward_dim: int, *, rngs: nnx.Rngs) -> None:
        self.norm1 = nnx.LayerNorm(embedding_dim, param_dtype=jnp.float64, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=heads_num,
            in_features=embedding_dim,
            qkv_features=embedding_dim,
            out_features=embedding_dim,
            param_dtype=jnp.float64,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(embedding_dim, param_dtype=jnp.float64, rngs=rngs)
        self.feed_forward = _FeedForward(embedding_dim, feed_forward_dim, rngs=rngs)

    def __call__(self, inputs: Array, mask: Array | None) -> Array:
        normed = self.norm1(inputs)
        attended = self.attention(normed, mask=mask)
        residual = inputs + attended
        return residual + self.feed_forward(self.norm2(residual))


class _Transformers(nnx.Module):
    """Stack of causal decoder units.

    In parallel mode a causal mask is built for the full sequence. In decode mode
    (single incremental token) the attention layers rely on their key/value cache
    and no mask is passed.
    """

    def __init__(
        self, embedding_dim: int, heads_num: int, feed_forward_dim: int, depth: int, *, rngs: nnx.Rngs
    ) -> None:
        self.units = nnx.List(
            [_DecoderUnit(embedding_dim, heads_num, feed_forward_dim, rngs=rngs) for _ in range(depth)]
        )

    def __call__(self, inputs: Array, *, decode: bool) -> Array:
        mask = None if decode else nnx.make_causal_mask(jnp.ones(inputs.shape[:2]))
        activation = inputs
        for unit in self.units:
            activation = unit(activation, mask)
        return activation


class _Tail(nnx.Module):
    """Output head: Linear -> GELU -> Linear producing amplitude and phase logits.

    The final layer is zero-initialised so the wave function starts near-uniform.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs) -> None:
        self.up = nnx.Linear(embedding_dim, hidden_dim, param_dtype=jnp.float64, rngs=rngs)
        self.down = nnx.Linear(
            hidden_dim, output_dim, kernel_init=nnx.initializers.zeros, param_dtype=jnp.float64, rngs=rngs
        )

    @nnx.jit
    def __call__(self, inputs: Array) -> Array:
        return self.down(jax.nn.gelu(self.up(inputs)))


class _TransformerWaveFunctionBase(nnx.Module):
    """Shared autoregressive machinery for the three transformer variants."""

    sites: int
    states_per_site: int
    _ordering: tuple[int, ...]
    _ordering_reversed: tuple[int, ...]

    def _build_networks(
        self,
        embedding_dim: int,
        heads_num: int,
        feed_forward_dim: int,
        depth: int,
        tail_hidden_dim: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.embedding = _PositionalEmbedding(self.sites, self.states_per_site, embedding_dim, rngs=rngs)
        self.transformers = _Transformers(embedding_dim, heads_num, feed_forward_dim, depth, rngs=rngs)
        self.tail = _Tail(embedding_dim, tail_hidden_dim, 2 * self.states_per_site, rngs=rngs)

    # ---- variant hooks ----

    def _config_to_site_values(self, configs: Array) -> Array:
        raise NotImplementedError

    def _site_values_to_config(self, site_values: Array) -> Array:
        raise NotImplementedError

    def _local_mask(self, prefix_values: Array, site_index: int) -> Array:
        raise NotImplementedError

    # ---- shared helpers ----

    @nnx.jit
    def _ordering_array(self) -> Array:
        return jnp.asarray(self._ordering, dtype=jnp.int32)

    @nnx.jit
    def _ordering_reversed_array(self) -> Array:
        return jnp.asarray(self._ordering_reversed, dtype=jnp.int32)

    @nnx.jit
    def _split_tail(self, hidden: Array) -> tuple[Array, Array]:
        """Split a tail activation into ``(raw_amplitude, phase)`` over the state axis."""
        tail = self.tail(hidden)
        return tail[..., : self.states_per_site], tail[..., self.states_per_site :]

    @functools.partial(nnx.jit, static_argnums=(3,))
    def _conditional_log_amplitude(self, raw_amplitude: Array, prefix_values: Array, site_index: int) -> Array:
        """Normalise a raw amplitude slice ``[batch, states]`` under the particle-number mask."""
        mask = self._local_mask(prefix_values, site_index)
        return normalize_log_amplitude(apply_mask(raw_amplitude, mask), axis=-1)

    @nnx.jit
    def __call__(self, configs: Array) -> Array:
        site_values = self._config_to_site_values(configs)
        batch_size = site_values.shape[0]
        batch_indices = jnp.arange(batch_size)

        # Causal input stream: BOS token followed by all but the last site.
        bos = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        tokens = bos if self.sites == 1 else jnp.concatenate([bos, site_values[:, :-1]], axis=1)

        embedded = self.embedding(tokens, position=0)
        hidden = self.transformers(embedded, decode=False)
        raw_amplitude, phase = self._split_tail(hidden)

        total_log_amplitude = jnp.zeros((batch_size,), dtype=jnp.float64)
        total_phase = jnp.zeros((batch_size,), dtype=jnp.float64)
        for site_index in range(self.sites):
            conditional = self._conditional_log_amplitude(
                raw_amplitude[:, site_index], site_values[:, :site_index], site_index
            )
            chosen = site_values[:, site_index]
            total_log_amplitude = total_log_amplitude + conditional[batch_indices, chosen]
            total_phase = total_phase + phase[:, site_index][batch_indices, chosen]

        return jnp.exp(total_log_amplitude + 1j * total_phase)

    # ---- incremental decoding ----

    def _init_decode_cache(self, batch_size: int) -> None:
        """Enable decode mode and allocate key/value caches for a full sequence."""
        self.transformers.set_attributes(decode=True)
        for unit in self.transformers.units:
            unit.attention.init_cache((batch_size, self.sites, self.embedding_dim), dtype=jnp.float64)

    def _disable_decode(self) -> None:
        self.transformers.set_attributes(decode=False)

    def _reorder_cache(self, parents: Array) -> None:
        """Reorder every attention key/value cache along the batch axis by ``parents``."""
        for unit in self.transformers.units:
            attention = unit.attention
            cached_key = attention.cached_key
            cached_value = attention.cached_value
            assert cached_key is not None
            assert cached_value is not None
            attention.cached_key = nnx.Cache(cached_key[...][parents])
            attention.cached_value = nnx.Cache(cached_value[...][parents])

    @functools.partial(nnx.jit, static_argnums=(2,))
    def _decode_step(self, token: Array, site_index: int) -> tuple[Array, Array]:
        """Feed one token ``[batch, 1]`` at ``site_index``; return ``(raw_amplitude, phase)`` ``[batch, states]``.

        JIT-compiled per ``site_index`` (static). The key/value cache is updated in place;
        ``nnx.jit`` threads the cache state through so the mutation is preserved across steps.
        """
        embedded = self.embedding(token, position=site_index)
        hidden = self.transformers(embedded, decode=True)
        raw_amplitude, phase = self._split_tail(hidden)
        return raw_amplitude[:, 0], phase[:, 0]

    def generate(self, batch_size: int, *, key: Array) -> tuple[Array, Array, Array]:
        self._init_decode_cache(batch_size)
        site_values = jnp.zeros((batch_size, 0), dtype=jnp.int32)
        token = jnp.zeros((batch_size, 1), dtype=jnp.int32)  # BOS
        for site_index in range(self.sites):
            raw_amplitude, _ = self._decode_step(token, site_index)
            conditional = self._conditional_log_amplitude(raw_amplitude, site_values, site_index)
            key, subkey = jax.random.split(key)
            choice = sample_step(conditional, subkey).astype(jnp.int32)
            site_values = jnp.concatenate([site_values, choice[:, None]], axis=1)
            token = choice[:, None]
        self._disable_decode()

        configs = self._site_values_to_config(site_values)
        unique_configs, counts = jnp.unique(configs, axis=0, return_counts=True)
        psi = self(unique_configs)
        return unique_configs, psi, counts

    def generate_unique(self, batch_size: int, *, key: Array) -> tuple[Array, Array]:
        beam_width = batch_size
        self._init_decode_cache(beam_width)

        beam_values = jnp.zeros((beam_width, 0), dtype=jnp.int32)
        beam_log_prob = jnp.full((beam_width,), 0.0, dtype=jnp.float64)
        beam_perturbed = jnp.full((beam_width,), 0.0, dtype=jnp.float64)
        beam_valid = jnp.arange(beam_width) == 0
        token = jnp.zeros((beam_width, 1), dtype=jnp.int32)  # BOS for every beam slot

        states = self.states_per_site
        for site_index in range(self.sites):
            raw_amplitude, _ = self._decode_step(token, site_index)
            conditional = self._conditional_log_amplitude(raw_amplitude, beam_values, site_index)
            key, subkey = jax.random.split(key)
            child_log_prob, child_perturbed, child_valid = gumbel_topk_step(
                beam_log_prob, beam_perturbed, beam_valid, 2.0 * conditional, subkey
            )

            tiled_prefix = jnp.broadcast_to(beam_values[:, None, :], (beam_width, states, site_index)).reshape(
                beam_width * states, site_index
            )
            next_states = jnp.broadcast_to(jnp.arange(states, dtype=jnp.int32)[None, :], (beam_width, states)).reshape(
                beam_width * states, 1
            )
            candidates = jnp.concatenate([tiled_prefix, next_states], axis=1)

            flat_perturbed = child_perturbed.reshape(-1)
            selected = jnp.argsort(flat_perturbed)[::-1][:beam_width]
            parents = (selected // states).astype(jnp.int32)

            beam_values = candidates[selected]
            beam_log_prob = child_log_prob.reshape(-1)[selected]
            beam_perturbed = flat_perturbed[selected]
            beam_valid = child_valid.reshape(-1)[selected]
            self._reorder_cache(parents)
            token = beam_values[:, site_index : site_index + 1]
        self._disable_decode()

        configs = self._site_values_to_config(beam_values)
        configs = configs[jnp.asarray(beam_valid)]
        psi = self(configs)
        return configs, psi


class WaveFunctionNormal(_TransformerWaveFunctionBase):
    """Transformer wave function without any particle-number conservation."""

    def __init__(
        self,
        *,
        sites: int,
        physical_dim: int,
        embedding_dim: int,
        heads_num: int,
        feed_forward_dim: int,
        depth: int,
        tail_hidden_dim: int,
        ordering: int | list[int] = 1,
        rngs: nnx.Rngs,
    ) -> None:
        self.sites = sites
        self.physical_dim = physical_dim
        self.states_per_site = physical_dim
        self._bit_size = _bit_size_for(physical_dim)
        self._ordering = _resolve_ordering(ordering, sites)
        self._ordering_reversed = _invert_permutation(self._ordering)
        self._build_networks(embedding_dim, heads_num, feed_forward_dim, depth, tail_hidden_dim, rngs)

    @nnx.jit
    def _config_to_site_values(self, configs: Array) -> Array:
        site_values = unpack_int(configs, size=self._bit_size, last_dim=self.sites).astype(jnp.int32)
        return site_values[:, self._ordering_reversed_array()]

    @nnx.jit
    def _site_values_to_config(self, site_values: Array) -> Array:
        user_order = site_values[:, self._ordering_array()]
        return pack_int(user_order.astype(jnp.uint8), size=self._bit_size)

    @functools.partial(nnx.jit, static_argnums=(2,))
    def _local_mask(self, prefix_values: Array, site_index: int) -> Array:
        batch_size = prefix_values.shape[0]
        return jnp.ones((batch_size, self.states_per_site), dtype=bool)


class WaveFunctionElectron(_TransformerWaveFunctionBase):
    """Transformer wave function conserving the total electron number."""

    def __init__(
        self,
        *,
        sites: int,
        electrons: int,
        embedding_dim: int,
        heads_num: int,
        feed_forward_dim: int,
        depth: int,
        tail_hidden_dim: int,
        ordering: int | list[int] = 1,
        rngs: nnx.Rngs,
    ) -> None:
        self.sites = sites
        self.electrons = electrons
        self.states_per_site = 2
        self._ordering = _resolve_ordering(ordering, sites)
        self._ordering_reversed = _invert_permutation(self._ordering)
        self._build_networks(embedding_dim, heads_num, feed_forward_dim, depth, tail_hidden_dim, rngs)

    @nnx.jit
    def _config_to_site_values(self, configs: Array) -> Array:
        site_values = unpack_int(configs, size=1, last_dim=self.sites).astype(jnp.int32)
        return site_values[:, self._ordering_reversed_array()]

    @nnx.jit
    def _site_values_to_config(self, site_values: Array) -> Array:
        user_order = site_values[:, self._ordering_array()]
        return pack_int(user_order.astype(jnp.uint8), size=1)

    @functools.partial(nnx.jit, static_argnums=(2,))
    def _local_mask(self, prefix_values: Array, site_index: int) -> Array:
        electron_count = jnp.sum(prefix_values, axis=1)
        sites_filled = jnp.asarray(site_index)
        return mask_electron(electron_count, sites_filled, self.sites, self.electrons)


class WaveFunctionElectronUpDown(_TransformerWaveFunctionBase):
    """Transformer wave function conserving spin-up and spin-down electron numbers.

    Each site is a pair of qubits (spin-up, spin-down) encoded as ``up * 2 + down``.
    """

    def __init__(
        self,
        *,
        double_sites: int,
        spin_up: int,
        spin_down: int,
        embedding_dim: int,
        heads_num: int,
        feed_forward_dim: int,
        depth: int,
        tail_hidden_dim: int,
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
        self._ordering = _resolve_ordering(ordering, self.sites)
        self._ordering_reversed = _invert_permutation(self._ordering)
        self._build_networks(embedding_dim, heads_num, feed_forward_dim, depth, tail_hidden_dim, rngs)

    @nnx.jit
    def _config_to_site_values(self, configs: Array) -> Array:
        qubits = unpack_int(configs, size=1, last_dim=self.double_sites).astype(jnp.int32)
        pairs = qubits.reshape(qubits.shape[0], self.sites, 2)
        site_values = pairs[:, :, 0] * 2 + pairs[:, :, 1]
        return site_values[:, self._ordering_reversed_array()]

    @nnx.jit
    def _site_values_to_config(self, site_values: Array) -> Array:
        user_order = site_values[:, self._ordering_array()]
        up = user_order // 2
        down = user_order % 2
        qubits = jnp.stack([up, down], axis=-1).reshape(user_order.shape[0], self.double_sites)
        return pack_int(qubits.astype(jnp.uint8), size=1)

    @functools.partial(nnx.jit, static_argnums=(2,))
    def _local_mask(self, prefix_values: Array, site_index: int) -> Array:
        up_count = jnp.sum(prefix_values // 2, axis=1)
        down_count = jnp.sum(prefix_values % 2, axis=1)
        sites_filled = jnp.asarray(site_index)
        mask = mask_electron_up_down(up_count, down_count, sites_filled, self.sites, self.spin_up, self.spin_down)
        return mask.reshape(mask.shape[0], 4)
