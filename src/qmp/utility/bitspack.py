"""JAX bit-packing utilities.

Combines multiple int1 / int2 / int4 / int8 values into single bytes and back.
The bit layout is LSB-first: the first element along the packed axis occupies
the least significant bits of the resulting byte.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array

_VALID_SIZES = (1, 2, 4, 8)


@functools.partial(jax.jit, static_argnames=("size",))
def pack_int(array: Array, size: int) -> Array:
    """Combine multiple small integers along the last dimension into bytes.

    Parameters
    ----------
    array : Array
        A uint8 array of shape ``[..., last_dim]``.
    size : int
        Bit width of each packed integer, one of ``{1, 2, 4, 8}``.

    Returns
    -------
    Array
        A uint8 array of shape ``[..., ceil(last_dim / (8 // size))]``.
    """
    if array.dtype != jnp.uint8:
        raise ValueError(f"pack_int expects uint8 input, got {array.dtype}.")
    if size not in _VALID_SIZES:
        raise ValueError(f"size must be one of {_VALID_SIZES}, got {size}.")

    elements_per_byte = 8 // size
    leading_shape = array.shape[:-1]
    last_dim = array.shape[-1]

    remainder = last_dim % elements_per_byte
    if remainder != 0:
        padding = elements_per_byte - remainder
        pad_width = [(0, 0)] * (array.ndim - 1) + [(0, padding)]
        array = jnp.pad(array, pad_width)
        last_dim = last_dim + padding

    num_bytes = last_dim // elements_per_byte
    grouped = array.reshape(*leading_shape, num_bytes, elements_per_byte)

    shifts = jnp.arange(0, 8, size, dtype=jnp.uint8)
    packed = jnp.sum(grouped << shifts, axis=-1, dtype=jnp.uint8)
    return packed


@functools.partial(jax.jit, static_argnames=("size", "last_dim"))
def unpack_int(array: Array, size: int, last_dim: int) -> Array:
    """Unpack bytes into multiple small integers along a new last dimension.

    Inverse of :func:`pack_int`.

    Parameters
    ----------
    array : Array
        A uint8 array of packed bytes with shape ``[..., num_bytes]``.
    size : int
        Bit width of each packed integer, one of ``{1, 2, 4, 8}``.
    last_dim : int
        The original last-dimension length prior to packing. Trailing padding
        elements beyond this length are dropped.

    Returns
    -------
    Array
        A uint8 array of shape ``[..., last_dim]``.
    """
    if array.dtype != jnp.uint8:
        raise ValueError(f"unpack_int expects uint8 input, got {array.dtype}.")
    if size not in _VALID_SIZES:
        raise ValueError(f"size must be one of {_VALID_SIZES}, got {size}.")

    shifts = jnp.arange(0, 8, size, dtype=jnp.uint8)
    value_mask = jnp.uint8((1 << size) - 1)

    unpacked = (array[..., None] >> shifts) & value_mask

    leading_shape = unpacked.shape[:-2]
    full_last_dim = unpacked.shape[-2] * unpacked.shape[-1]
    unpacked = unpacked.reshape(*leading_shape, full_last_dim)
    return unpacked[..., :last_dim]
