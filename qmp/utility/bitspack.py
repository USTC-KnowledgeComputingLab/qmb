"""
This module provides functions to combine multiple int1, int2, int4 or int8 values into a single byte.
"""

import torch


@torch.jit.script
def pack_int(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """
    Combines multiple small int values into a single byte.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor with shape ... * last_dimension.
    size : 1 | 2 | 4 | 8
        The size of the int values in bits.

    Returns
    -------
    torch.Tensor
        The packed tensor with shape ... * ((last_dimension + elements_per_byte - 1) // elements_per_byte).
    """
    assert tensor.dtype == torch.uint8
    assert size in [1, 2, 4, 8]

    # Get the shape of the input tensor
    shape = list(tensor.shape)
    # Get the size of the last dimension
    last_dim = shape.pop()
    # Get the number of elements per byte
    elements_per_byte = 8 // size

    # Calculate how many zeros need to be padded
    if last_dim % elements_per_byte != 0:
        pad = elements_per_byte - (last_dim % elements_per_byte)
        tensor = torch.nn.functional.pad(tensor, (0, pad))
        last_dim = last_dim + pad

    # Calculate the number of groups
    num_bytes = last_dim // elements_per_byte

    # Reshape the tensor to (..., num_bytes, elements_per_byte)
    shape.append(num_bytes)
    shape.append(elements_per_byte)
    tensor = tensor.view(shape)

    # Define the shifts and calculate the byte value for each group
    shifts = torch.arange(0, 8, size, device=tensor.device, dtype=torch.uint8)
    packed = torch.sum(tensor << shifts, dim=-1, dtype=torch.uint8)

    assert packed.dtype == torch.uint8
    return packed


@torch.jit.script
def unpack_int(tensor: torch.Tensor, size: int, last_dim: int) -> torch.Tensor:
    """
    Unpacks bytes into multiple small int values based on the specified size.

    Parameters
    ----------
    tensor : torch.Tensor
        The packed tensor with bytes.
    size : 1 | 2 | 4 | 8
        The size of the int values in bits.
    last_dim : int
        The original size of the last dimension before packing.

    Returns
    -------
    torch.Tensor
        The unpacked tensor with shape ... * last_dim.
    """
    assert tensor.dtype == torch.uint8
    assert size in [1, 2, 4, 8]

    # Define the shifts and mask
    shifts = torch.arange(0, 8, size, device=tensor.device, dtype=torch.uint8)
    mask = torch.tensor((1 << size) - 1, device=tensor.device, dtype=torch.uint8)

    # Calculate the unpacked values
    unpacked = torch.bitwise_and(torch.bitwise_right_shift(tensor.unsqueeze(-1), shifts), mask)

    # Reshape the unpacked tensor to the original shape
    shape = list(unpacked.shape)
    shape.append(shape.pop() * shape.pop())
    unpacked = unpacked.view(shape)[..., :last_dim]

    assert unpacked.dtype == torch.uint8
    return unpacked
