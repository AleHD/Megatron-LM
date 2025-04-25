from typing import Optional

import torch
from torch import nn

from megatron.core.transformer import TransformerConfig


class DynamicTanh(nn.Module):
    def __init__(self,
        config: TransformerConfig,
        hidden_size: int,
        eps: Optional[float] = None,  # Unused. Added to match LayerNorm interface.
        init_value: Optional[float] = None,  # gamma init value.
        location: Optional[None] = None,
    ):

        super().__init__()

        self.config = config
        self.hidden_size = hidden_size
        self.init_value = init_value
        self.location = location

        self.alpha = nn.Parameter(torch.empty(1))
        self.weight = nn.Parameter(torch.empty(hidden_size))
        if self.config.dyt_bias:
            self.beta = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        # Determine the alpha init value.
        if self.location is None or self.location == "qk":
            alpha0 = self.config.dyt_alpha_init
        elif self.location == "attention":
            alpha0 = self.config.dyt_alpha_init_attention
        elif self.location == "other":
            alpha0 = self.config.dyt_alpha_init_other
        else:
            raise ValueError(f"Unknown location {self.location}")

        nn.init.constant_(self.alpha, alpha0)
        if self.init_value is None:
            nn.init.ones_(self.weight)
        else:
            nn.init.constant_(self.weight, self.init_value)
        if self.config.dyt_bias:
            nn.init.zeros_(self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.dyt_bias:
            return dyt_bias(x, self.weight, self.alpha, self.beta)
        return dyt(x, self.weight, self.alpha)
        #x = self.weight*torch.tanh(self.alpha*x)
        #if self.config.dyt_bias:
        #    return x + self.beta
        #return x


@torch.compile
def dyt(x: torch.Tensor, weight: torch.Tensor, alpha: torch.Tensor):
    return weight*torch.tanh(alpha*x)

@torch.compile
def dyt_bias(x: torch.Tensor, weight: torch.Tensor, alpha: torch.Tensor, bias: torch.Tensor):
    return weight*torch.tanh(alpha*x) + bias
