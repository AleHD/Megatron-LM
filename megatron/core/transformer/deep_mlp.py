from typing import Callable

import torch
from torch import nn

# TODO: Make it a megatron module.
class DeepMLP(nn.Module):
    def __init__(self, size_in: int, size_hidden: int, size_out: int, n_layers: int,
                 activation_factory: Callable[[], Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        assert n_layers > 0
        layers = [nn.Sequential(nn.RMSNorm(size_in), nn.Linear(size_in, size_hidden, bias=False), activation_factory())]
        for _ in range(n_layers - 1):
            layers.append(nn.Sequential(nn.RMSNorm(size_hidden), nn.Linear(size_hidden, size_hidden, bias=False), activation_factory()))
        layers.append(nn.Sequential(nn.RMSNorm(size_hidden), nn.Linear(size_hidden, size_out, bias=False)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l_no, layer in enumerate(self.layers):
            y = layer(x)
            if x.size() == y.size() and l_no + 1 < len(self.layers):
                x = x + y
            else:
                x = y
        return x

