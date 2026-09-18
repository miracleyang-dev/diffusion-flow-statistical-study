"""Fast smoke tests."""

import torch

from diffusion_flow_study.config import ExperimentConfig
from diffusion_flow_study.data import sample_gaussian_mixture, sample_swiss_roll
from diffusion_flow_study.experiment import run_experiment
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


def test_swiss_roll_supports_three_dimensional_methods() -> None:
    generator = torch.Generator().manual_seed(11)
    clean = sample_swiss_roll(8, generator=generator)
    assert clean.shape == (8, 3)
    assert torch.isfinite(clean).all()
    assert sliced_wasserstein(clean, clean) < 1e-6

    methods = [
        DDPM(TimeMLP(hidden_dim=16, data_dim=3), diffusion_steps=4, data_dim=3),
        VPScoreSDE(TimeMLP(hidden_dim=16, data_dim=3), data_dim=3),
        FlowMatching(TimeMLP(hidden_dim=16, data_dim=3), data_dim=3),
    ]
    for method in methods:
        assert torch.isfinite(method.loss(clean))
        sample_steps = 4 if isinstance(method, DDPM) else 2
        samples = method.sample(8, sample_steps, generator)
        assert samples.shape == (8, 3)
        assert torch.isfinite(samples).all()


def test_swiss_roll_experiment_writes_artifacts(tmp_path) -> None:
    payload = run_experiment(
        ExperimentConfig(
            dataset="swiss_roll",
            train_steps=1,
            batch_size=8,
            sample_count=8,
            sample_steps=2,
            ddpm_train_steps=2,
            hidden_dim=8,
            output_dir=str(tmp_path),
        )
    )
    assert payload["config"]["dataset"] == "swiss_roll"
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics.png").exists()
    assert (tmp_path / "samples.png").exists()


def test_mnist_vector_path_supports_three_methods() -> None:
    generator = torch.Generator().manual_seed(13)
    clean = torch.rand(8, 28 * 28, generator=generator)
    methods = [
        DDPM(TimeMLP(hidden_dim=8, data_dim=28 * 28), diffusion_steps=4, data_dim=28 * 28),
        VPScoreSDE(TimeMLP(hidden_dim=8, data_dim=28 * 28), data_dim=28 * 28),
        FlowMatching(TimeMLP(hidden_dim=8, data_dim=28 * 28), data_dim=28 * 28),
    ]
    for method in methods:
        assert torch.isfinite(method.loss(clean))
        sample_steps = 4 if isinstance(method, DDPM) else 2
        samples = method.sample(8, sample_steps, generator)
        assert samples.shape == (8, 28 * 28)
        assert torch.isfinite(samples).all()
