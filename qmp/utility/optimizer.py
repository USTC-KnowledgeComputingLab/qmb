"""
This module provides tools for PyTorch optimizers.
"""

import typing
import torch


def _migrate_tensor(tensor: torch.Tensor, device: torch.device) -> None:
    """
    Migrates the tensor to the specified device.
    """
    tensor.data = tensor.data.to(device=device)
    if tensor.grad is not None:
        tensor.grad.data = tensor.grad.data.to(device=device)


def _migrate_param(param: object, device: torch.device) -> None:
    """
    Migrates the parameter to the specified device.
    """
    if isinstance(param, torch.Tensor):
        _migrate_tensor(param, device)
    elif isinstance(param, list):
        for subparam in param:
            _migrate_param(subparam, device)
    elif isinstance(param, dict):
        for subparam in param.values():
            _migrate_param(subparam, device)
    elif isinstance(param, int | float | complex):
        pass
    else:
        raise ValueError(f"Unexpected parameter type: {type(param)}")


def _migrate_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """
    Migrates the optimizer to the device of the model parameters.
    """
    device: torch.device = optimizer.param_groups[0]["params"][0].device
    _migrate_param(optimizer.state, device)


def initialize_optimizer(
    params: typing.Iterable[torch.Tensor],
    *,
    use_lbfgs: bool,
    learning_rate: float,
    new_opt: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    state_dict: typing.Any = None,
) -> torch.optim.Optimizer:
    """
    Initialize an optimizer.

    Parameters
    ----------
    params : typing.Iterable[torch.Tensor]
        The parameters to be optimized.
    use_lbfgs : bool
        Whether to use L-BFGS as the optimizer.
    learning_rate : float
        The initial learning rate.
    new_opt : bool, default=True
        Whether to create a new optimizer or use the given optimizer, by default True.
    optimizer : torch.optim.Optimizer, optional
        The optimizer to be used if new_opt is False, by default None.
    state_dict : typing.Any, optional
        The state dictionary of the optimizer to be loaded, by default None.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer.
    """
    if new_opt or optimizer is None:
        if use_lbfgs:
            optimizer = torch.optim.LBFGS(params, lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(params, lr=learning_rate)
    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
        _migrate_optimizer(optimizer)
    return optimizer


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
