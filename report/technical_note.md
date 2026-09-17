# DDPM, Score-SDE, and Flow Matching: A Statistical Comparison

> Working outline for the final 6–10 page note. Numerical claims remain blank
> until the multi-seed benchmark is frozen.

## Abstract

We compare discrete denoising diffusion, continuous-time score-based diffusion,
and conditional flow matching under controlled data, architecture, optimization,
and evaluation settings. The study emphasizes distributional error,
quality–compute trade-offs, and run-to-run stability.

## 1. Research questions

1. Which objective best recovers separated modes under a matched training budget?
2. How rapidly does each sampler improve as the number of function evaluations grows?
3. How do noise schedules interact with low-dimensional or curved support?
4. What fidelity cost is induced by simple reward guidance?

## 2. Methods

### 2.1 DDPM

Define the discrete forward process, epsilon-prediction objective, linear/cosine
schedules, and ancestral reverse transition.

### 2.2 Score-SDE

Define the variance-preserving SDE, marginal perturbation kernel, weighted
denoising score-matching objective, and Euler–Maruyama reverse solver.

### 2.3 Flow Matching

Define the Gaussian-to-data probability path, conditional velocity target, and
Euler integration of the learned ODE.

## 3. Experimental design

- Data: eight-mode Gaussian mixture, Swiss roll, and MNIST.
- Controls: common capacity, optimizer family, batch size, update budget, and seeds.
- Primary metrics: RBF MMD², sliced Wasserstein-2, and MNIST FID/KID.
- Compute metrics: sampling steps/NFE and wall-clock time.
- Stability: mean ± standard deviation, bootstrap confidence intervals, and failure rate.
- Statistical protocol: at least five seeds; paired seed-level comparisons where possible.

## 4. Results

### 4.1 Gaussian mixture

Insert target/sample panels, metric table, and quality-versus-NFE curve.

### 4.2 Swiss roll and low-dimensional structure

Report ambient distributional metrics together with distance-to-manifold and
neighborhood-preservation diagnostics.

### 4.3 MNIST

Report image metrics, class coverage, throughput, and representative samples.

## 5. Ablations and guided sampling

Compare linear and cosine schedules. Add a differentiable reward favoring a
chosen mixture mode or digit class, then report reward improvement against MMD,
Wasserstein/FID, and mode coverage.

## 6. Discussion and limitations

Separate optimization effects from numerical-solver effects; discuss metric
sensitivity, small-model limitations, timing comparability, and the gap between
toy geometry and image generation.

## 7. Conclusion

Summarize empirically supported trade-offs only after results are frozen.

## Reproducibility appendix

Record commit hash, dependency lockfile, hardware, commands, seeds, exact
configurations, and per-seed raw results.
