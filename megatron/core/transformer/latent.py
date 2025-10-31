from abc import ABC, abstractmethod
from typing import Optional

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
        self.reset_parameters()

    def forward(self, hidden_states: torch.Tensor, latent_states: torch.Tensor, **kw) -> torch.Tensor:
        return self.linear(torch.cat([hidden_states, latent_states], dim=-1))

    def reset_parameters(self):
        weight = self.linear.weight.data
        alpha = self.config.linear_latent_adapter_alpha
        # TODO: It would be nice if we can conserve V[output] after our init tweak.
        with torch.no_grad():
            eye = torch.eye(self.config.hidden_size, dtype=weight.dtype, device=weight.device)
            weight[:, :self.config.hidden_size] = alpha*weight[:, :self.config.hidden_size] + (1-alpha)*eye                    
            weight[:, self.config.hidden_size:] = alpha*weight[:, self.config.hidden_size:] + (1-alpha)*0.0




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
        tau = torch.normal(torch.log(torch.tensor(self.config.n_recurrences - 1.0)) - 0.5*std**2, std)
        return (torch.poisson(torch.exp(tau)) + 1).item()


#= Latent Refiner =#
class LatentRefiner(nn.Module, ABC):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.requires_attn_scores = False

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,  # (S, B, H).
        latent_states: torch.Tensor,  # (S, B, H).
        attention_mask: torch.Tensor, # (B, 1, S, S).
        rotary_pos_embed: torch.Tensor,  # (B, 1, 1, freq).
        attn_scores: Optional[torch.Tensor] = None, # (B, nheads, S, S).
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def rejoin(self, hidden_states, latent_states) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class NoLatentRefiner(LatentRefiner):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.refined = False

    def forward(self, hidden_states, latent_states, attention_mask, rotary_pos_embed, attn_scores=None):
        assert not self.refined
        self.refined = True
        self.attention_mask = attention_mask
        self.rotary_pos_embed = rotary_pos_embed
        return hidden_states, latent_states, attention_mask, rotary_pos_embed

    def rejoin(self, hidden_states, latent_states):
        assert self.refined
        self.refined = False
        return hidden_states, latent_states, self.attention_mask, self.rotary_pos_embed


class TopkSeqLatentRefiner(LatentRefiner):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.refined = False
        self.requires_attn_scores = True

    def forward(self, hidden_states, latent_states, attention_mask, rotary_pos_embed, attn_scores=None):
        assert attn_scores is not None
        assert not self.refined
        self.refined = True
        self.hidden_states = hidden_states
        self.latent_states = latent_states
        self.attention_mask = attention_mask
        self.rotary_pos_embed = rotary_pos_embed
        self.attn_scores = attn_scores  

        attn_scores = torch.sum(attn_scores, dim=1)  # (B, S, S).



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
