"""Fast smoke tests."""

import torch

from diffusion_flow_study.data import sample_gaussian_mixture
from diffusion_flow_study.methods import DDPM, FlowMatching, VPScoreSDE
from diffusion_flow_study.metrics import rbf_mmd, sliced_wasserstein
from diffusion_flow_study.models import TimeMLP


def test_data_and_metrics() -> None:
    x = sample_gaussian_mixture(64)
    assert x.shape == (64, 2)
    assert rbf_mmd(x, x) < 1e-6
    assert sliced_wasserstein(x, x) < 1e-6


def test_methods_produce_finite_samples() -> None:
    generator = torch.Generator().manual_seed(7)
    methods = [
        DDPM(TimeMLP(hidden_dim=16), diffusion_steps=4),
        VPScoreSDE(TimeMLP(hidden_dim=16)),
        FlowMatching(TimeMLP(hidden_dim=16)),
    ]
    clean = sample_gaussian_mixture(8, generator=generator)
    for method in methods:
        assert torch.isfinite(method.loss(clean))
        sample_steps = 4 if isinstance(method, DDPM) else 2
        samples = method.sample(8, sample_steps, generator)
        assert samples.shape == (8, 2)
        assert torch.isfinite(samples).all()
