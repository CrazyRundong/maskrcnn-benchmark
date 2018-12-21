# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__all__ = ["enable_quant_conv", ]


def nudge_min_max(k, lb, ub):
    ub = torch.max(ub, lb + 1e-2)
    lb = torch.min(torch.tensor(0., device=lb.device), lb)
    ub = torch.max(torch.tensor(0., device=ub.device), ub)
    n = 2 ** k - 1
    delta = (ub - lb) / n
    z_idx = torch.round(torch.abs(lb) / delta)
    lb = - z_idx * delta
    ub = (n - z_idx) * delta

    return lb, ub, z_idx, delta


def linear_quant(x, k, lb=None, ub=None):
    if k >= 32:
        return x
    elif k == 1:
        _sign = Sign.apply
        scale = x.detach().abs().mean()
        qx = scale * _sign(x)
        return qx

    with torch.no_grad():
        if lb is None or ub is None:
            lb = x.min()
            ub = x.max()
        lb, ub, z_idx, delta = nudge_min_max(k, lb, ub)

    _round = Round.apply
    x = torch.clamp(x, lb, ub)
    qx = _round((x - lb) / delta) * delta + lb

    return qx


def enable_quant_conv(k, lb=None, ub=None):
    def _quant_conv2d(input, weight, bias, stride, padding, dilation, groups):
        input = linear_quant(input, k, lb, ub)
        weight = linear_quant(weight, k)
        return _conv2d(input, weight, bias, stride, padding, dilation, groups)

    _conv2d = F.conv2d
    F.conv2d = _quant_conv2d


class Round(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, dy):
        return dy


class Sign(Function):
    @staticmethod
    def forward(x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, dy):
        return dy
