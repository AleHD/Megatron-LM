from typing import Optional

import torch
from torch import nn


class LayerScale(nn.Module):
    def __init__(self, hidden_size: Optional[int] = None, initial_value: float = 1.0, device=None, dtype=None):
        super().__init__()
        if hidden_size is None:
            self.weight = torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype))
        else:
            self.weight = torch.nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.initial_value = initial_value
        self.reset_parameters()

    def reset_parameters(self):
        # pass
        nn.init.constant_(self.weight, self.initial_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return self.initial_value*x
        return self.weight * x
