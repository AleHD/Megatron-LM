from abc import ABC, abstractmethod

from megatron.core.transformer.transformer_config import TransformerConfig
import torch
from torch import nn

#= Adapters =#
class LatentAdapter(nn.Module, ABC):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, latent_states: torch.Tensor, **kw) -> torch.Tensor:
        pass


class NullLatentAdapter(LatentAdapter):
    def forward(self, hidden_states: torch.Tensor, latent_states: torch.Tensor, **kw) -> torch.Tensor:
        return latent_states


class LinearLatentAdapter(LatentAdapter):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, latent_states: torch.Tensor, **kw) -> torch.Tensor:
        return self.linear(torch.cat([hidden_states, latent_states], dim=-1))



#= Initializers =#
class LatentInitializer(nn.Module, ABC):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, **kw) -> torch.Tensor:
        pass


class IdentityInitializer(LatentInitializer):
    def forward(self, hidden_states: torch.Tensor, **kw) -> torch.Tensor:
        return hidden_states


class TruncNormInitializer(LatentInitializer):
    def forward(self, hidden_states: torch.Tensor, **kw) -> torch.Tensor:
        return torch.clip(self.config.latent_init_std*torch.randn_like(hidden_states), -3*self.config.latent_init_std, 3*self.config.latent_init_std)



#= LatentTimes =#
class LatentTimes(nn.Module, ABC):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, **kw) -> int:
        pass

    def early_exit(self, latent_states: torch.Tensor, previous_latent_states: torch.Tensor) -> bool:
        return False


class ConstantTimes(LatentTimes):
    def forward(self, **kw) -> int:
        return self.config.n_recurrences


class PoissonTimes(LatentTimes):
    def forward(self, **kw) -> int:
        std = torch.tensor(0.5)
        tau = torch.normal(torch.log(torch.tensor(self.config.n_recurrences - 1.0)) - 0.5*std**0.5, std)
        return (torch.poisson(torch.exp(tau)) + 1).item()


ADAPTERS = {
    "none": NullLatentAdapter,
    "linear": LinearLatentAdapter,
}

INITIALIZERS = {
    "identity": IdentityInitializer,
    "truncnorm": TruncNormInitializer,
}

TIMES = {
    "constant": ConstantTimes,
    "poisson": PoissonTimes,
}
