import math

import torch
import torch.nn as nn


class QValue(nn.Module):
    """MLP Q-function Q(x, a) -> scalar.

    ``x`` and ``a`` are concatenated and passed through hidden layers of
    Linear -> LayerNorm -> activation (LayerNorm optional, activation
    configurable, Tanh by default), then a linear head to a single value.
    The same class serves the action critic Q(s, a) and the latent-noise
    critic Q(s, w): only ``action_dim`` differs.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        activation: type[nn.Module] = nn.Tanh,
        layer_norm: bool = True,
        orthogonal_init: bool = False,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim + action_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

        if orthogonal_init:
            for m in self.trunk:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.head.weight, gain=1.0)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return Q(x, a) with shape (...,)."""
        return self.head(self.trunk(torch.cat([x, a], dim=-1))).squeeze(-1)
