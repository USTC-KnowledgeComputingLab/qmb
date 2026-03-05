"""
Tests for qmp.utility.bitspack pack_int and unpack_int.
"""

import torch

from qmp.utility.bitspack import pack_int, unpack_int


class TestPackUnpackIntSize1:
    """pack_int / unpack_int round-trip for 1-bit integers."""

    def test_exact_multiple_of_elements_per_byte(self) -> None:
        """16 elements pack into 2 bytes and unpack back without loss."""
        tensor = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1], dtype=torch.uint8)
        packed = pack_int(tensor, 1)
        unpacked = unpack_int(packed, 1, tensor.shape[-1])
        assert torch.all(unpacked == tensor)

    def test_with_padding(self) -> None:
        """15 elements require padding to 16; unpacking strips the pad and recovers the original."""
        tensor = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], dtype=torch.uint8)
        packed = pack_int(tensor, 1)
        unpacked = unpack_int(packed, 1, tensor.shape[-1])
        assert torch.all(unpacked == tensor)

    def test_multi_dimensional(self) -> None:
        """4×4 tensor packs each row independently and unpacks back correctly."""
        tensor = torch.tensor([[1, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.uint8)
        packed = pack_int(tensor, 1)
        unpacked = unpack_int(packed, 1, tensor.shape[-1])
        assert torch.all(unpacked == tensor)

    def test_large_random_tensor(self) -> None:
        """1000 random 1-bit values survive a pack/unpack round-trip."""
        tensor = torch.randint(0, 2, (1000,), dtype=torch.uint8)
        packed = pack_int(tensor, 1)
        unpacked = unpack_int(packed, 1, tensor.shape[-1])
        assert torch.all(unpacked == tensor)


class TestPackUnpackIntSize2:
    """pack_int / unpack_int round-trip for 2-bit integers."""

    def test_exact_multiple_of_elements_per_byte(self) -> None:
        """16 elements (values 0–3) pack into 4 bytes and unpack back without loss."""
        tensor = torch.tensor([3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0], dtype=torch.uint8)
        packed = pack_int(tensor, 2)
        unpacked = unpack_int(packed, 2, tensor.shape[-1])
        assert torch.all(unpacked == tensor)

    def test_with_padding(self) -> None:
        """14 elements require padding to 16; unpacking strips the pad and recovers the original."""
        tensor = torch.tensor([3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2], dtype=torch.uint8)
        packed = pack_int(tensor, 2)
        unpacked = unpack_int(packed, 2, tensor.shape[-1])
        assert torch.all(unpacked == tensor)

    def test_large_random_tensor(self) -> None:
        """1000 random 2-bit values survive a pack/unpack round-trip."""
        tensor = torch.randint(0, 4, (1000,), dtype=torch.uint8)
        packed = pack_int(tensor, 2)
        unpacked = unpack_int(packed, 2, tensor.shape[-1])
        assert torch.all(unpacked == tensor)


class TestPackUnpackIntSize4:
    """pack_int / unpack_int round-trip for 4-bit integers."""

    def test_exact_multiple_of_elements_per_byte(self) -> None:
        """16 elements (values 0–15) pack into 8 bytes and unpack back without loss."""
        tensor = torch.tensor([15, 10, 5, 0, 15, 10, 5, 0, 15, 10, 5, 0, 15, 10, 5, 0], dtype=torch.uint8)
        packed = pack_int(tensor, 4)
        unpacked = unpack_int(packed, 4, tensor.shape[-1])
        assert torch.all(unpacked == tensor)

    def test_with_padding(self) -> None:
        """14 elements require padding to 16; unpacking strips the pad and recovers the original."""
        tensor = torch.tensor([15, 10, 5, 0, 15, 10, 5, 0, 15, 10, 5, 0, 15, 10], dtype=torch.uint8)
        packed = pack_int(tensor, 4)
        unpacked = unpack_int(packed, 4, tensor.shape[-1])
        assert torch.all(unpacked == tensor)

    def test_large_random_tensor(self) -> None:
        """1000 random 4-bit values survive a pack/unpack round-trip."""
        tensor = torch.randint(0, 16, (1000,), dtype=torch.uint8)
        packed = pack_int(tensor, 4)
        unpacked = unpack_int(packed, 4, tensor.shape[-1])
        assert torch.all(unpacked == tensor)


class TestPackUnpackIntSize8:
    """pack_int / unpack_int round-trip for 8-bit integers (full byte, no packing)."""

    def test_full_byte_values(self) -> None:
        """16 arbitrary byte values are unchanged by a size-8 pack/unpack round-trip."""
        tensor = torch.tensor([15, 210, 5, 123, 151, 130, 5, 0, 15, 10, 75, 0, 15, 10, 5, 0], dtype=torch.uint8)
        packed = pack_int(tensor, 8)
        unpacked = unpack_int(packed, 8, tensor.shape[-1])
        assert torch.all(unpacked == tensor)
