"""Tests for JAX bit-packing utilities.

Covers: round-trip for all sizes, padding boundaries, explicit bit-layout,
multi-dimensional inputs, dtype preservation.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from qmp.utility.bitspack import pack_int, unpack_int


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_round_trip_aligned(size: int) -> None:
    """pack then unpack recovers the original array when last_dim is aligned."""
    elements_per_byte = 8 // size
    last_dim = elements_per_byte * 3
    max_value = (1 << size) - 1
    values = (jnp.arange(last_dim, dtype=jnp.uint8) % (max_value + 1)).reshape(1, last_dim)
    packed = pack_int(values, size)
    unpacked = unpack_int(packed, size, last_dim)
    assert jnp.array_equal(unpacked, values)


@pytest.mark.parametrize("size", [1, 2, 4])
def test_round_trip_unaligned(size: int) -> None:
    """Round-trip works when last_dim is not a multiple of elements_per_byte."""
    elements_per_byte = 8 // size
    last_dim = elements_per_byte * 2 + 1
    max_value = (1 << size) - 1
    values = (jnp.arange(last_dim, dtype=jnp.uint8) % (max_value + 1)).reshape(1, last_dim)
    packed = pack_int(values, size)
    unpacked = unpack_int(packed, size, last_dim)
    assert jnp.array_equal(unpacked, values)


def test_pack_byte_count_aligned() -> None:
    """Packed size for size=1 with 8 bits is one byte."""
    values = jnp.array([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=jnp.uint8)
    packed = pack_int(values, size=1)
    assert packed.shape == (1, 1)


def test_pack_byte_count_unaligned() -> None:
    """Packed size rounds up when last_dim is not aligned."""
    values = jnp.array([[1, 0, 1]], dtype=jnp.uint8)
    packed = pack_int(values, size=1)
    assert packed.shape == (1, 1)


def test_bit_layout_lsb_first() -> None:
    """size=1 packs the first element into the least significant bit."""
    values = jnp.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=jnp.uint8)
    packed = pack_int(values, size=1)
    assert int(packed[0, 0]) == 0b00000001

    values2 = jnp.array([[0, 0, 0, 0, 0, 0, 0, 1]], dtype=jnp.uint8)
    packed2 = pack_int(values2, size=1)
    assert int(packed2[0, 0]) == 0b10000000


def test_bit_layout_size2() -> None:
    """size=2 packs four 2-bit values LSB-first within a byte."""
    values = jnp.array([[0b01, 0b10, 0b11, 0b00]], dtype=jnp.uint8)
    packed = pack_int(values, size=2)
    # element0 -> bits 0-1, element1 -> bits 2-3, element2 -> bits 4-5, element3 -> bits 6-7
    expected = 0b01 | (0b10 << 2) | (0b11 << 4) | (0b00 << 6)
    assert int(packed[0, 0]) == expected


def test_size8_identity() -> None:
    """size=8 is a passthrough (one element per byte)."""
    values = jnp.array([[7, 42, 255, 0]], dtype=jnp.uint8)
    packed = pack_int(values, size=8)
    assert jnp.array_equal(packed, values)
    unpacked = unpack_int(packed, size=8, last_dim=4)
    assert jnp.array_equal(unpacked, values)


def test_multi_dimensional() -> None:
    """Packing operates on the last dimension only, preserving leading dims."""
    values = jnp.array(
        [
            [[1, 0, 1, 1], [0, 0, 1, 0]],
            [[1, 1, 1, 1], [0, 1, 0, 1]],
        ],
        dtype=jnp.uint8,
    )
    packed = pack_int(values, size=1)
    assert packed.shape == (2, 2, 1)
    unpacked = unpack_int(packed, size=1, last_dim=4)
    assert jnp.array_equal(unpacked, values)


def test_dtype_preserved() -> None:
    """Both operations preserve uint8 dtype."""
    values = jnp.array([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=jnp.uint8)
    packed = pack_int(values, size=1)
    assert packed.dtype == jnp.uint8
    unpacked = unpack_int(packed, size=1, last_dim=8)
    assert unpacked.dtype == jnp.uint8


def test_unpack_truncates_padding() -> None:
    """Unpacking with a smaller last_dim drops the padding elements."""
    values = jnp.array([[1, 1, 1]], dtype=jnp.uint8)
    packed = pack_int(values, size=1)
    unpacked = unpack_int(packed, size=1, last_dim=3)
    assert unpacked.shape == (1, 3)
    assert jnp.array_equal(unpacked, values)


# ---- error paths ----


def test_pack_rejects_non_uint8() -> None:
    values = jnp.array([[1, 0, 1, 0]], dtype=jnp.int32)
    with pytest.raises(ValueError, match="uint8"):
        pack_int(values, size=1)


def test_unpack_rejects_non_uint8() -> None:
    packed = jnp.array([[5]], dtype=jnp.int32)
    with pytest.raises(ValueError, match="uint8"):
        unpack_int(packed, size=1, last_dim=8)


@pytest.mark.parametrize("size", [0, 3, 5, 7, 16])
def test_pack_rejects_invalid_size(size: int) -> None:
    values = jnp.array([[1, 0, 1, 0]], dtype=jnp.uint8)
    with pytest.raises(ValueError, match="size must be one of"):
        pack_int(values, size=size)


@pytest.mark.parametrize("size", [0, 3, 5, 7, 16])
def test_unpack_rejects_invalid_size(size: int) -> None:
    packed = jnp.array([[5]], dtype=jnp.uint8)
    with pytest.raises(ValueError, match="size must be one of"):
        unpack_int(packed, size=size, last_dim=4)


def test_max_values_round_trip() -> None:
    """Maximum representable value per size survives a round trip."""
    for size in (1, 2, 4, 8):
        max_value = (1 << size) - 1
        elements_per_byte = 8 // size
        values = jnp.full((1, elements_per_byte * 2), max_value, dtype=jnp.uint8)
        unpacked = unpack_int(pack_int(values, size), size, last_dim=elements_per_byte * 2)
        assert jnp.array_equal(unpacked, values)
