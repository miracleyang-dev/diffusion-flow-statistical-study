"""Sample-quality metrics implemented directly in PyTorch."""

import torch
from torch import Tensor


def _median_bandwidth(x: Tensor, y: Tensor, max_points: int = 1_000) -> Tensor:
    z = torch.cat((x[:max_points], y[:max_points]), dim=0)
    distances = torch.pdist(z).square()
    return distances.median().clamp_min(1e-6)


def rbf_mmd(x: Tensor, y: Tensor, *, max_points: int | None = None) -> float:
    """Biased RBF-kernel maximum mean discrepancy (squared)."""
    if max_points is not None:
        x = x[:max_points]
        y = y[:max_points]
    bandwidth = _median_bandwidth(x, y)
    k_xx = torch.exp(-torch.cdist(x, x).square() / bandwidth)
    k_yy = torch.exp(-torch.cdist(y, y).square() / bandwidth)
    k_xy = torch.exp(-torch.cdist(x, y).square() / bandwidth)
    return float((k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()).clamp_min(0.0))


def sliced_wasserstein(
    x: Tensor, y: Tensor, *, projections: int = 256, seed: int = 0
) -> float:
    """Monte Carlo estimate of the 2-Wasserstein distance over 1D slices."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be rank-2 tensors with the same feature dimension")
    n = min(x.shape[0], y.shape[0])
    generator = torch.Generator(device=x.device).manual_seed(seed)
    directions = torch.randn(x.shape[1], projections, device=x.device, generator=generator)
    directions = directions / directions.norm(dim=0, keepdim=True)
    x_proj = (x[:n] @ directions).sort(dim=0).values
    y_proj = (y[:n] @ directions).sort(dim=0).values
    return float((x_proj - y_proj).square().mean().sqrt())
