import math
import torch


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        match_rms: float = 0.2,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        nesterov: bool = True,
    ):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            match_rms=match_rms,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            nesterov=nesterov,
            step=0
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1
            step = group["step"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            match_rms = group["match_rms"]
            ns_steps = group["ns_steps"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            nesterov = group["nesterov"]
            is_embedding_or_output = group["is_embedding_or_output"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                use_muon = p.ndim >= 2 and not is_embedding_or_output

                # Init state.
                if use_muon and "muon_buffer" not in state:
                    state["muon_buffer"] = torch.zeros_like(p)
                elif not use_muon and "adamw_exp_avg" not in state:
                    state["adamw_exp_avg"] = torch.zeros_like(g)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(g)

                # Compute muon.
                if use_muon:
                    buf = state["muon_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if nesterov else buf
                    update = newton_schulz(g, steps=ns_steps)*math.sqrt(max(p.size(-1), p.size(-2)))*match_rms
                    p.data.mul_(1 - lr*weight_decay)
                    p.data.add_(update, alpha=-lr)
                # Compute adam.
                else:
                    buf1 = state["adamw_exp_avg"]
                    buf2 = state["adamw_exp_avg_sq"]
                    buf1.lerp_(g, 1-beta1)
                    buf2.lerp_(g.square(), 1-beta2)
                    g = buf1/(eps + buf2.sqrt())
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    scale = bias_correction1/bias_correction2**0.5
                    p.data.mul_(1 - lr*weight_decay)
                    p.data.add_(g, alpha=-lr/scale)


# from https://github.com/KellerJordan/Muon/tree/master.
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B =  b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X
