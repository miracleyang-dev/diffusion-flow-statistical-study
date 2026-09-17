"""End-to-end benchmark runner with metrics and publication-ready figures."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import ExperimentConfig
from .data import sample_gaussian_mixture
from .methods import DDPM, FlowMatching, GenerativeMethod, VPScoreSDE
from .metrics import rbf_mmd, sliced_wasserstein
from .models import TimeMLP


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_methods(config: ExperimentConfig) -> dict[str, GenerativeMethod]:
    def model() -> TimeMLP:
        return TimeMLP(hidden_dim=config.hidden_dim)

    return {
        "DDPM": DDPM(model(), diffusion_steps=config.ddpm_train_steps, device=config.device),
        "Score-SDE": VPScoreSDE(
            model(), beta_min=config.beta_min, beta_max=config.beta_max, device=config.device
        ),
        "Flow Matching": FlowMatching(model(), device=config.device),
    }


def _plot_samples(reference: torch.Tensor, samples: dict[str, torch.Tensor], path: Path) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.35), sharex=True, sharey=True)
    panels = {"Target": reference, **samples}
    for axis, (name, points) in zip(axes, panels.items(), strict=True):
        array = points.numpy()
        axis.scatter(array[:, 0], array[:, 1], s=3, alpha=0.45, linewidths=0)
        axis.set_title(name)
        axis.set_aspect("equal")
        axis.set_xlim(-5.3, 5.3)
        axis.set_ylim(-5.3, 5.3)
        axis.grid(alpha=0.15)
    figure.suptitle("Eight-mode Gaussian mixture: target and generated samples", y=1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_metrics(results: dict[str, dict[str, float | int | list[float]]], path: Path) -> None:
    names = list(results)
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    specifications = (
        ("mmd_rbf", "RBF MMD²", "#35618f"),
        ("sliced_wasserstein", "Sliced Wasserstein", "#9a4f3d"),
        ("sampling_seconds", "Sampling time (s)", "#477a5b"),
    )
    for axis, (key, label, color) in zip(axes, specifications, strict=True):
        values = [float(results[name][key]) for name in names]
        axis.bar(names, values, color=color, width=0.65)
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Train all methods under a shared budget, evaluate, and save artifacts."""
    _seed_everything(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    data_generator = torch.Generator(device=device).manual_seed(config.seed + 10)

    def data_sampler(n: int) -> torch.Tensor:
        return sample_gaussian_mixture(n, device=device, generator=data_generator)

    reference = data_sampler(config.sample_count).cpu()
    generated: dict[str, torch.Tensor] = {}
    metrics: dict[str, dict[str, float | int | list[float]]] = {}

    for method_index, (name, method) in enumerate(_build_methods(config).items()):
        _seed_everything(config.seed + method_index)
        started = time.perf_counter()
        losses = method.fit(
            data_sampler,
            train_steps=config.train_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
        )
        training_seconds = time.perf_counter() - started
        sampling_steps = config.ddpm_train_steps if name == "DDPM" else config.sample_steps
        sample_generator = torch.Generator(device=device).manual_seed(
            config.seed + 100 + method_index
        )
        started = time.perf_counter()
        sample = method.sample(config.sample_count, sampling_steps, sample_generator).cpu()
        sampling_seconds = time.perf_counter() - started
        generated[name] = sample
        metrics[name] = {
            "mmd_rbf": rbf_mmd(reference, sample),
            "sliced_wasserstein": sliced_wasserstein(reference, sample, seed=config.seed),
            "sampling_steps": sampling_steps,
            "training_seconds": training_seconds,
            "sampling_seconds": sampling_seconds,
            "final_logged_loss": losses[-1],
            "loss_trace": losses,
        }

    payload: dict[str, object] = {"config": asdict(config), "results": metrics}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot_samples(reference, generated, output / "samples.png")
    _plot_metrics(metrics, output / "metrics.png")
    return payload
