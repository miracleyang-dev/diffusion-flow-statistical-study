"""Synthetic two-dimensional datasets."""

import math

import torch
from torch import Tensor


def sample_gaussian_mixture(
    n: int,
    *,
    modes: int = 8,
    radius: float = 4.0,
    std: float = 0.25,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample an equally weighted ring of isotropic Gaussian components."""
    component = torch.randint(modes, (n,), device=device, generator=generator)
    angles = 2.0 * math.pi * component.float() / modes
    centers = radius * torch.stack((angles.cos(), angles.sin()), dim=1)
    noise = torch.randn(n, 2, device=device, generator=generator)
    return centers + std * noise
