# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import torch
# import torch.nn as nn
import jittor as jt
from jittor import nn

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def execute(self, x: jt.array) -> jt.array:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        # self.weight = nn.Parameter(torch.ones(num_channels))
        # self.bias = nn.Parameter(torch.zeros(num_channels))
        self.weight = jt.Var(jt.ones(num_channels))
        self.bias = jt.Var(jt.zeros(num_channels))
        self.eps = eps

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    def execute(self, x: jt.array) -> jt.array:
        # u = x.mean(1, keepdim=True)
        u = jt.mean(x, dim=1, keepdims=True)
        # s = (x - u).pow(2).mean(1, keepdim=True)
        s = jt.mean((x - u).pow(2), dim=1, keepdims=True)
        x = (x - u) / jt.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x