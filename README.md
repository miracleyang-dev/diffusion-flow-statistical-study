# Diffusion–Flow Statistical Study

A controlled empirical study of **DDPM**, **Score-based SDEs**, and **Flow
Matching**. The project asks how objective choice and sampling dynamics affect
sample quality, computational cost, and stability when architecture, data, and
training budget are held fixed.

The first milestone is a fully reproducible two-dimensional benchmark on an
eight-mode Gaussian mixture. Planned extensions cover Swiss roll, MNIST,
noise-schedule ablations, low-dimensional structure, and reward-guided
sampling. The accompanying technical note will turn the results into a compact
6–10 page statistical comparison rather than a collection of isolated demos.

## What is implemented

- **DDPM** with a linear discrete variance schedule and ancestral sampling.
- **VP Score-SDE** with continuous-time denoising score matching and a reverse-SDE sampler.
- **Conditional Flow Matching** with straight Gaussian-to-data paths and Euler ODE sampling.
- A shared time-conditioned MLP architecture and matched optimization budget.
- RBF MMD, sliced Wasserstein distance, sampling steps, training/sampling wall time,
  loss traces, and explicit non-finite-loss checks.
- Reproducible JSON results and two publication-ready figures.

## Quick start

Prerequisites: Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run diffusion-flow-study
```

For a short smoke experiment:

```bash
uv run diffusion-flow-study \
  --train-steps 200 \
  --sample-count 500 \
  --sample-steps 50 \
  --ddpm-steps 50
```

The run creates:

```text
artifacts/gaussian_mixture/
├── metrics.json     # configuration, losses, quality, steps, and timing
├── metrics.png      # side-by-side metric comparison
└── samples.png      # target and generated distributions
```

Useful options:

```bash
uv run diffusion-flow-study --help
uv run diffusion-flow-study --device cuda
uv run diffusion-flow-study --seed 123 --output-dir artifacts/seed_123
```

## Experimental contract

The MVP deliberately controls the following factors:

| Factor | Shared setting |
|---|---|
| Data | Eight equally weighted isotropic Gaussian modes on a ring |
| Network | Three-layer SiLU MLP with fixed Fourier time features |
| Optimizer | AdamW with gradient clipping |
| Training data | Fresh samples from the same population distribution |
| Evaluation | Same reference size and random seed policy |

Sampling steps are reported rather than forced to be identical: DDPM follows
its discrete training schedule, while Score-SDE and Flow Matching expose an
independent numerical integration budget. This makes quality-versus-NFE curves
a first-class follow-up experiment.

## Metrics

- **RBF MMD²** measures kernel mean-embedding discrepancy; bandwidth is selected
  by the pooled median heuristic.
- **Sliced Wasserstein-2** averages one-dimensional transport discrepancies over
  random projections.
- **Runtime and steps** expose computational trade-offs that distributional
  metrics alone hide.
- **Stability** currently includes deterministic seeding, gradient clipping, and
  non-finite-loss failure checks. Multi-seed mean, standard deviation, and
  failure-rate reporting are the next milestone.

## Repository layout

```text
src/diffusion_flow_study/
├── cli.py            # command-line entry point
├── config.py         # experiment configuration
├── data.py           # synthetic target distributions
├── experiment.py     # training, evaluation, JSON, and figures
├── methods.py        # DDPM, Score-SDE, and Flow Matching
├── metrics.py        # MMD and sliced Wasserstein
└── models.py         # shared time-conditioned MLP
report/
└── technical_note.md # 6–10 page note outline
tests/
└── test_smoke.py
```

## Roadmap

- [x] Two-dimensional Gaussian-mixture MVP.
- [ ] Multi-seed runs with confidence intervals and failure rates.
- [ ] Quality-versus-sampling-step Pareto curves.
- [ ] Swiss-roll geometry and intrinsic-dimension diagnostics.
- [ ] Linear, cosine, and learned/noise-schedule ablations.
- [ ] MNIST convolutional backbone with FID/KID-style evaluation.
- [ ] Simple reward-guided sampling and reward–fidelity trade-off curves.
- [ ] Final 6–10 page technical note with frozen configurations and results.

## Reproducibility notes

Timing results depend on hardware and should be compared only within a single
machine/runtime configuration. The current code seeds Python, NumPy, PyTorch,
training data, and sampling streams, but exact GPU determinism is not claimed.
Commit the generated `uv.lock` file whenever dependencies change.

## License

MIT
