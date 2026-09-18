"""Time-conditioned neural networks."""

import math

import torch
from torch import Tensor, nn


class FourierTimeEmbedding(nn.Module):
    """Fixed Fourier features for scalar diffusion/flow time."""

    frequencies: Tensor

    def __init__(self, dim: int = 32, scale: float = 16.0) -> None:
        super().__init__()
        frequencies = torch.exp(torch.linspace(0.0, math.log(scale), dim // 2))
        self.register_buffer("frequencies", frequencies)

    def forward(self, t: Tensor) -> Tensor:
        phase = 2.0 * math.pi * t[:, None] * self.frequencies[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class TimeMLP(nn.Module):
    """Compact shared architecture used by all three methods."""

    data_dim: int

    def __init__(self, hidden_dim: int = 128, time_dim: int = 32, data_dim: int = 2) -> None:
        super().__init__()
        if data_dim <= 0:
            raise ValueError("data_dim must be positive")
        self.time_embedding = FourierTimeEmbedding(time_dim)
        self.data_dim = data_dim
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.data_dim:
            raise ValueError(
                f"expected x with shape (batch, {self.data_dim}), got {tuple(x.shape)}"
            )
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        return self.net(torch.cat((x, self.time_embedding(t)), dim=-1))
