"""Experiment configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Small defaults keep the benchmark runnable on a laptop."""

    seed: int = 42
    device: str = "cpu"
    train_steps: int = 5_000
    batch_size: int = 512
    learning_rate: float = 2e-3
    hidden_dim: int = 128
    sample_count: int = 2_000
    sample_steps: int = 100
    ddpm_train_steps: int = 200
    beta_min: float = 0.1
    beta_max: float = 20.0
    output_dir: str = "artifacts/gaussian_mixture"
