"""Time-conditioned neural networks."""

import math

import torch
from torch import Tensor, nn


class FourierTimeEmbedding(nn.Module):
    """Fixed Fourier features for scalar diffusion/flow time."""

    def __init__(self, dim: int = 32, scale: float = 16.0) -> None:
        super().__init__()
        frequencies = torch.exp(torch.linspace(0.0, math.log(scale), dim // 2))
        self.register_buffer("frequencies", frequencies)

    def forward(self, t: Tensor) -> Tensor:
        phase = 2.0 * math.pi * t[:, None] * self.frequencies[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class TimeMLP(nn.Module):
    """Compact shared architecture used by all three methods."""

    def __init__(self, hidden_dim: int = 128, time_dim: int = 32) -> None:
        super().__init__()
        self.time_embedding = FourierTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        return self.net(torch.cat((x, self.time_embedding(t)), dim=-1))
