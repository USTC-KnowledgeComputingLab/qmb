"""
This file contains various loss functions used in the other script.

These functions help calculate the difference between the target wave function and the current state wave function.
"""

import math
import torch


@torch.jit.script
def _scaled_abs(s_abs: torch.Tensor, min_magnitude: float) -> torch.Tensor:
    s_large = torch.log(s_abs)
    s_small = (s_abs - min_magnitude) / min_magnitude + math.log(min_magnitude)
    return torch.where(s_abs > min_magnitude, s_large, s_small)


@torch.jit.script
def _scaled_angle(scale: torch.Tensor, min_magnitude: float) -> torch.Tensor:
    return 1 / (1 + min_magnitude / scale)


# 损失函数目录:
# log: 基于对数差的损失函数。
# sum_reweighted_log: 基于对数差并按幅度之和重加权的损失函数。
# sum_filtered_log: 基于对数差并按幅度之和过滤的损失函数。
# sum_filtered_scaled_log: 基于缩放对数差并按幅度之和过滤的损失函数。
# sum_reweighted_angle_log: 仅对角度部分按幅度之和重加权的对数损失函数。
# sum_filtered_angle_log: 仅对角度部分按幅度之和过滤的对数损失函数。
# sum_filtered_angle_scaled_log: 仅对角度部分按幅度之和过滤的缩放对数损失函数。
# direct: 直接计算波函数差异的损失函数。


@torch.jit.script
def log(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the log of the current state wave function and the target wave function.
    """
    s_abs = torch.sqrt(s.real**2 + s.imag**2)
    t_abs = torch.sqrt(t.real**2 + t.imag**2)

    s_angle = torch.atan2(s.imag, s.real)
    t_angle = torch.atan2(t.imag, t.real)

    s_magnitude = torch.log(s_abs)
    t_magnitude = torch.log(t_abs)

    error_real = (s_magnitude - t_magnitude) / (2 * torch.pi)
    error_imag = (s_angle - t_angle) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()

    loss = error_real**2 + error_imag**2
    return loss.mean()


@torch.jit.script
def sum_reweighted_log(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the log of the current state wave function and the target wave function,
    but reweighted by the sum of the abs of the current state wave function and the target wave function.
    """
    s_abs = torch.sqrt(s.real**2 + s.imag**2)
    t_abs = torch.sqrt(t.real**2 + t.imag**2)

    s_angle = torch.atan2(s.imag, s.real)
    t_angle = torch.atan2(t.imag, t.real)

    s_magnitude = torch.log(s_abs)
    t_magnitude = torch.log(t_abs)

    error_real = (s_magnitude - t_magnitude) / (2 * torch.pi)
    error_imag = (s_angle - t_angle) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()

    loss = error_real**2 + error_imag**2
    loss = loss * (t_abs + s_abs)
    return loss.mean()


@torch.jit.script
def sum_filtered_log(s: torch.Tensor, t: torch.Tensor, min_magnitude: float = 1e-10) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the log of the current state wave function and the target wave function,
    but filtered by the sum of the abs of the current state wave function and the target wave function.
    """
    s_abs = torch.sqrt(s.real**2 + s.imag**2)
    t_abs = torch.sqrt(t.real**2 + t.imag**2)

    s_angle = torch.atan2(s.imag, s.real)
    t_angle = torch.atan2(t.imag, t.real)

    s_magnitude = torch.log(s_abs)
    t_magnitude = torch.log(t_abs)

    error_real = (s_magnitude - t_magnitude) / (2 * torch.pi)
    error_imag = (s_angle - t_angle) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()

    loss = error_real**2 + error_imag**2
    loss = loss * _scaled_angle(t_abs + s_abs, min_magnitude)
    return loss.mean()


@torch.jit.script
def sum_filtered_scaled_log(s: torch.Tensor, t: torch.Tensor, min_magnitude: float = 1e-10) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the scaled log of the current state wave function and the target wave function,
    but filtered by the sum of the abs of the current state wave function and the target wave function.
    """
    s_abs = torch.sqrt(s.real**2 + s.imag**2)
    t_abs = torch.sqrt(t.real**2 + t.imag**2)

    s_angle = torch.atan2(s.imag, s.real)
    t_angle = torch.atan2(t.imag, t.real)

    s_magnitude = _scaled_abs(s_abs, min_magnitude)
    t_magnitude = _scaled_abs(t_abs, min_magnitude)

    error_real = (s_magnitude - t_magnitude) / (2 * torch.pi)
    error_imag = (s_angle - t_angle) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()

    loss = error_real**2 + error_imag**2
    loss = loss * _scaled_angle(t_abs + s_abs, min_magnitude)
    return loss.mean()


@torch.jit.script
def sum_reweighted_angle_log(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the log of the current state wave function and the target wave function,
    but angle only reweighted by the sum of the abs of the current state wave function and the target wave function.
    """
    s_abs = torch.sqrt(s.real**2 + s.imag**2)
    t_abs = torch.sqrt(t.real**2 + t.imag**2)

    s_angle = torch.atan2(s.imag, s.real)
    t_angle = torch.atan2(t.imag, t.real)

    s_magnitude = torch.log(s_abs)
    t_magnitude = torch.log(t_abs)

    error_real = (s_magnitude - t_magnitude) / (2 * torch.pi)
    error_imag = (s_angle - t_angle) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()

    error_imag = error_imag * (t_abs + s_abs)
    loss = error_real**2 + error_imag**2
    return loss.mean()


@torch.jit.script
def sum_filtered_angle_log(s: torch.Tensor, t: torch.Tensor, min_magnitude: float = 1e-10) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the log of the current state wave function and the target wave function,
    but angle only filtered by the sum of the abs of the current state wave function and the target wave function.
    """
    s_abs = torch.sqrt(s.real**2 + s.imag**2)
    t_abs = torch.sqrt(t.real**2 + t.imag**2)

    s_angle = torch.atan2(s.imag, s.real)
    t_angle = torch.atan2(t.imag, t.real)

    s_magnitude = torch.log(s_abs)
    t_magnitude = torch.log(t_abs)

    error_real = (s_magnitude - t_magnitude) / (2 * torch.pi)
    error_imag = (s_angle - t_angle) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()

    error_imag = error_imag * _scaled_angle(t_abs + s_abs, min_magnitude)
    loss = error_real**2 + error_imag**2
    return loss.mean()


@torch.jit.script
def sum_filtered_angle_scaled_log(s: torch.Tensor, t: torch.Tensor, min_magnitude: float = 1e-10) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the scaled log of the current state wave function and the target wave function,
    but angle only filtered by the sum of the abs of the current state wave function and the target wave function.
    """
    s_abs = torch.sqrt(s.real**2 + s.imag**2)
    t_abs = torch.sqrt(t.real**2 + t.imag**2)

    s_angle = torch.atan2(s.imag, s.real)
    t_angle = torch.atan2(t.imag, t.real)

    s_magnitude = _scaled_abs(s_abs, min_magnitude)
    t_magnitude = _scaled_abs(t_abs, min_magnitude)

    error_real = (s_magnitude - t_magnitude) / (2 * torch.pi)
    error_imag = (s_angle - t_angle) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()

    error_imag = error_imag * _scaled_angle(t_abs + s_abs, min_magnitude)
    loss = error_real**2 + error_imag**2
    return loss.mean()


@torch.jit.script
def direct(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the current state wave function and the target wave function directly.
    """
    error = s - t
    loss = error.real**2 + error.imag**2
    return loss.mean()
