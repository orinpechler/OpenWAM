import math

import torch
import torch.nn as nn


class LatentActor(nn.Module):
    """MLP that outputs the mean and log-std of a diagonal Gaussian.

    Each hidden layer is Linear -> LayerNorm -> activation (LayerNorm optional,
    activation configurable, Tanh by default); the mean and log-std come
    from two separate linear heads, and log-std is clamped to
    [log_std_min, log_std_max]. ``sample`` draws with the reparameterization
    trick and, when ``set_range`` is True, squashes the sample with a final tanh
    and scales it into (-range_scale, range_scale).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        activation: type[nn.Module] = nn.Tanh,
        layer_norm: bool = True,
        set_range: bool = True,
        range_scale: float = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        orthogonal_init: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.set_range = set_range
        self.range_scale = range_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        if set_range and range_scale <= 0:
            raise ValueError("range_scale must be positive")

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, output_dim)
        self.log_std_head = nn.Linear(in_dim, output_dim)

        if orthogonal_init:
            # Orthogonal hidden layers; near-zero heads so the initial outputs are
            # mean ~ 0 and log_std ~ 0 (std ~ 1), i.e. roughly N(0, I).
            for m in self.trunk:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    nn.init.zeros_(m.bias)
            for head in (self.mean_head, self.log_std_head):
                nn.init.orthogonal_(head.weight, gain=0.01)
                nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std), each of shape (..., output_dim)."""
        h = self.trunk(x)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return self.mean_head(h), log_std

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample and its log-probability, shape (...,).

        z = mean + std * eps with eps ~ N(0, I). With ``set_range`` the sample is
        a = range_scale * tanh(z), and the log-probability includes the
        change-of-variables correction for that squashing:
            log p(a) = log N(z) - sum log(range_scale * (1 - tanh(z)^2)).
        """
        mean, log_std = self(x)
        eps = torch.randn_like(mean)
        z = mean + log_std.exp() * eps
        # log N(z; mean, std), summed over the output dimension.
        log_prob = (-0.5 * eps.pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        if not self.set_range:
            return z, log_prob

        # log(1 - tanh(z)^2) = 2 * (log 2 - z - softplus(-2z)), numerically stable.
        log_det = 2 * (math.log(2) - z - nn.functional.softplus(-2 * z)) + math.log(self.range_scale)
        return self.range_scale * torch.tanh(z), log_prob - log_det.sum(-1)

    def deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """The mean instead of a sample, squashed like ``sample`` when ``set_range``."""
        mean, _ = self(x)
        return self.range_scale * torch.tanh(mean) if self.set_range else mean
