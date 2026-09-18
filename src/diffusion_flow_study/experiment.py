"""End-to-end benchmark runner with metrics and publication-ready figures."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict, cast

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .config import ExperimentConfig
from .data import MNIST_IMAGE_SHAPE, dataset_dimension, sample_dataset
from .methods import DDPM, FlowMatching, GenerativeMethod, VPScoreSDE
from .metrics import rbf_mmd, sliced_wasserstein
from .models import TimeMLP


class MethodMetrics(TypedDict):
    """JSON-serializable metrics for one trained method."""

    mmd_rbf: float
    sliced_wasserstein: float
    sampling_steps: int
    training_seconds: float
    sampling_seconds: float
    final_logged_loss: float
    loss_trace: list[float]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_methods(config: ExperimentConfig, data_dim: int) -> dict[str, GenerativeMethod]:
    def model() -> TimeMLP:
        return TimeMLP(hidden_dim=config.hidden_dim, data_dim=data_dim)

    return {
        "DDPM": DDPM(
            model(),
            diffusion_steps=config.ddpm_train_steps,
            device=config.device,
            data_dim=data_dim,
        ),
        "Score-SDE": VPScoreSDE(
            model(),
            beta_min=config.beta_min,
            beta_max=config.beta_max,
            device=config.device,
            data_dim=data_dim,
        ),
        "Flow Matching": FlowMatching(model(), device=config.device, data_dim=data_dim),
    }


def _plot_samples(
    reference: torch.Tensor,
    samples: dict[str, torch.Tensor],
    path: Path,
    *,
    dataset: str,
) -> None:
    data_dim = reference.shape[1]
    if dataset == "mnist":
        rows, columns = 8, 8
        figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.35))
        panels = {"Target": reference, **samples}
        for axis, (name, points) in zip(axes, panels.items(), strict=True):
            images = torch.zeros(
                rows * columns,
                *MNIST_IMAGE_SHAPE,
                dtype=points.dtype,
                device=points.device,
            )
            count = min(points.shape[0], rows * columns)
            images[:count] = points[:count].reshape(-1, *MNIST_IMAGE_SHAPE)
            grid = images.reshape(rows, columns, *MNIST_IMAGE_SHAPE).permute(0, 2, 1, 3)
            grid = grid.reshape(rows * MNIST_IMAGE_SHAPE[0], columns * MNIST_IMAGE_SHAPE[1])
            axis.imshow(grid.detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(name)
            axis.axis("off")
    elif data_dim == 2:
        figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.35), sharex=True, sharey=True)
    elif data_dim == 3:
        figure = plt.figure(figsize=(13.2, 3.7))
        axes = [
            figure.add_subplot(1, 4, index + 1, projection="3d")
            for index in range(4)
        ]
    else:
        raise ValueError("sample plotting supports only two- or three-dimensional data")

    if dataset != "mnist":
        panels = {"Target": reference, **samples}
        for axis, (name, points) in zip(axes, panels.items(), strict=True):
            array = points.detach().cpu().numpy()
            if data_dim == 2:
                axis.scatter(array[:, 0], array[:, 1], s=3, alpha=0.45, linewidths=0)
            else:
                axis_3d = cast(Axes3D, axis)
                # The local Matplotlib stub types ``zs`` as a scalar, but the
                # runtime API accepts an array for the third coordinate.
                scatter_3d = cast(Callable[..., Any], axis_3d.scatter)
                scatter_3d(array[:, 0], array[:, 1], array[:, 2], s=3, alpha=0.45)
            axis.set_title(name)
            axis.grid(alpha=0.15)
            if data_dim == 3:
                cast(Axes3D, axis).view_init(elev=25, azim=-60)

    if data_dim == 2:
        for axis in axes:
            axis.set_aspect("equal")
            axis.set_xlim(-5.3, 5.3)
            axis.set_ylim(-5.3, 5.3)
    elif data_dim == 3:
        extent = max(float(reference.abs().amax().item()) * 1.05, 1.0)
        for axis in axes:
            axis.set_xlim(-extent, extent)
            axis.set_ylim(-extent, extent)
            axis.set_zlim(-extent, extent)
            axis.set_box_aspect((1, 1, 1))

    figure.suptitle(f"{dataset.replace('_', ' ').title()}: target and generated samples", y=1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_metrics(results: dict[str, MethodMetrics], path: Path) -> None:
    names = list(results)
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    specifications: tuple[tuple[str, str, str, Callable[[MethodMetrics], float]], ...] = (
        ("mmd_rbf", "RBF MMD²", "#35618f", lambda result: result["mmd_rbf"]),
        (
            "sliced_wasserstein",
            "Sliced Wasserstein",
            "#9a4f3d",
            lambda result: result["sliced_wasserstein"],
        ),
        (
            "sampling_seconds",
            "Sampling time (s)",
            "#477a5b",
            lambda result: result["sampling_seconds"],
        ),
    )
    for axis, (_, label, color, value_getter) in zip(axes, specifications, strict=True):
        values = [value_getter(results[name]) for name in names]
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
    data_dim = dataset_dimension(config.dataset)
    data_generator = torch.Generator(device=device).manual_seed(config.seed + 10)

    def data_sampler(n: int) -> torch.Tensor:
        return sample_dataset(
            config.dataset,
            n,
            data_dir=config.data_dir,
            device=device,
            generator=data_generator,
        )

    reference = data_sampler(config.sample_count).cpu()
    metric_count = config.metric_sample_count or config.sample_count
    metric_reference = reference[:metric_count]
    generated: dict[str, torch.Tensor] = {}
    metrics: dict[str, MethodMetrics] = {}

    for method_index, (name, method) in enumerate(_build_methods(config, data_dim).items()):
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
        metric_sample = sample[:metric_count]
        metrics[name] = {
            "mmd_rbf": rbf_mmd(metric_reference, metric_sample),
            "sliced_wasserstein": sliced_wasserstein(
                metric_reference, metric_sample, seed=config.seed
            ),
            "sampling_steps": sampling_steps,
            "training_seconds": training_seconds,
            "sampling_seconds": sampling_seconds,
            "final_logged_loss": losses[-1],
            "loss_trace": losses,
        }

    payload: dict[str, object] = {"config": asdict(config), "results": metrics}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot_samples(reference, generated, output / "samples.png", dataset=config.dataset)
    _plot_metrics(metrics, output / "metrics.png")
    return payload
