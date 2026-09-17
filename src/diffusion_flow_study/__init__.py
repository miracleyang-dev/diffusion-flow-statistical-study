"""Controlled benchmarks for diffusion and flow generative models."""

from .config import ExperimentConfig
from .experiment import run_experiment

__all__ = ["ExperimentConfig", "run_experiment"]
