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
class LatentMasker(nn.Module, ABC):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.requires_attn_scores = False

    @abstractmethod
    def forward(
        self,
        attn_scores: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:  # bool(B, 1, S, S).
        pass

    def aggregate_scores(self, agg: Optional[torch.Tensor], scores: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None



class NoLatentMasker(LatentMasker):
    def forward(self, attn_scores=None):
        return None


class TopkSeqLatentMasker(LatentMasker):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.requires_attn_scores = True

    # scores=(B, S, S).
    def forward(self, attn_scores=None):
        assert attn_scores is not None
        aggregated_scores = attn_scores
        #aggregated_scores = torch.sum(attn_scores, dim=(0, 2))  # (B, S, S).
        _, idx = torch.topk(aggregated_scores, self.config.latent_topk_masker_k, dim=2)
        B, S, _ = aggregated_scores.size()
        newmask = torch.ones(B, S, S, device=attn_scores.device, dtype=torch.bool)
        newmask = torch.scatter(newmask, 1, idx, torch.zeros_like(newmask))
        for i in range(self.config.latent_topk_masker_k + 1):  # No peaking into the future.
            newmask[:, i, i + 1:] = True
        return newmask[:, None, :, :]

    # scores=(B,nheads,S,S)
    def aggregate_scores(self, agg, scores):
        if scores is None:
            return agg
        scores = scores.detach()
        scores = torch.mean(scores, dim=1)/self.config.n_think_layers
        if agg is None:
            return scores
        return agg + scores
        

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

MASKERS = {
    "none": NoLatentMasker,
    "topk": TopkSeqLatentMasker,
}
