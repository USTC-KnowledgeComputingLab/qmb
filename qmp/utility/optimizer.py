"""
This module provides tools for PyTorch optimizers.
"""

import torch


def migrate_tensor(tensor: torch.Tensor, device: torch.device) -> None:
    """
    Migrates the tensor to the specified device.
    """
    tensor.data = tensor.data.to(device=device)
    if tensor.grad is not None:
        tensor.grad.data = tensor.grad.data.to(device=device)


def migrate_param(param: object, device: torch.device) -> None:
    """
    Migrates the parameter to the specified device.
    """
    if isinstance(param, torch.Tensor):
        migrate_tensor(param, device)
    elif isinstance(param, list):
        for subparam in param:
            migrate_param(subparam, device)
    elif isinstance(param, dict):
        for subparam in param.values():
            migrate_param(subparam, device)
    elif isinstance(param, int | float | complex):
        pass
    else:
        raise ValueError(f"Unexpected parameter type: {type(param)}")


def migrate_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """
    Migrates the optimizer to the device of the model parameters.
    """
    if not optimizer.param_groups:
        return
    # Assuming all params are on the same device or the first one is representative
    param_group = optimizer.param_groups[0]
    if not param_group["params"]:
        return
    device: torch.device = param_group["params"][0].device
    migrate_param(optimizer.state, device)


def scale_learning_rate(optimizer: torch.optim.Optimizer, scale: float) -> None:
    """
    Scales the learning rate of all parameter groups in the optimizer by a given factor.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate will be scaled.
    scale : float
        The factor by which the learning rate will be scaled.
    """
    for param in optimizer.param_groups:
        param["lr"] *= scale
