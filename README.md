# Diffusion–Flow Statistical Study / 扩散与流模型统计研究

A controlled empirical study of **DDPM**, **Score-based SDEs**, and **Flow
Matching**. The project asks how objective choice and sampling dynamics affect
sample quality, computational cost, and stability when architecture, data, and
training budget are held fixed.

The first milestone is a fully reproducible two-dimensional benchmark on an
eight-mode Gaussian mixture. Further extensions cover MNIST image modeling,
noise-schedule ablations, low-dimensional structure, and reward-guided
sampling. The accompanying technical note will turn the results into a compact
6–10 page statistical comparison rather than a collection of isolated demos.

本项目对 **DDPM**、**Score-based SDE** 与 **Flow Matching** 进行受控实证比较。在固定
网络结构、数据与训练预算的条件下，研究不同训练目标和采样动力学如何影响生成质量、
计算成本及稳定性。

首个里程碑是在八模态二维高斯混合分布上建立完整可复现实验。后续将扩展到 Swiss roll、
MNIST、噪声调度消融、低维结构与 reward-guided sampling。配套 technical note 将形成一份
6–10 页的统计比较，而非若干相互独立的演示。

## What is implemented / 已实现内容

- **DDPM** with a linear discrete variance schedule and ancestral sampling.
- **VP Score-SDE** with continuous-time denoising score matching and a reverse-SDE sampler.
- **Conditional Flow Matching** with straight Gaussian-to-data paths and Euler ODE sampling.
- A shared time-conditioned MLP architecture and matched optimization budget.
- RBF MMD, sliced Wasserstein distance, sampling steps, training/sampling wall time,
  loss traces, and explicit non-finite-loss checks.
- Reproducible JSON results and two publication-ready figures.
- A dimension-generalized benchmark path for the three-dimensional Swiss roll.
- An official IDX MNIST loader and flattened 784-dimensional baseline.

- 使用线性离散方差调度和祖先采样的 **DDPM**。
- 使用连续时间去噪 score matching 和反向 SDE 采样器的 **VP Score-SDE**。
- 使用高斯到数据直线路径及 Euler ODE 采样的 **Conditional Flow Matching**。
- 三种方法共享 time-conditioned MLP 和一致的优化预算。
- 提供 RBF MMD、Sliced Wasserstein、采样步数、训练/采样耗时、loss 轨迹及非有限值检查。
- 输出可复现的 JSON 结果和两张可用于技术报告的实验图。

## Quick start / 快速开始

Prerequisites: Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).

前置要求：Python 3.10+ 与 [`uv`](https://docs.astral.sh/uv/)。

```bash
uv sync --dev
uv run diffusion-flow-study
```

For a short smoke experiment / 运行短版冒烟实验：

```bash
uv run diffusion-flow-study \
  --train-steps 200 \
  --sample-count 500 \
  --sample-steps 50 \
  --ddpm-steps 50
```

Run the next benchmark dataset, the three-dimensional Swiss roll:

```bash
uv run diffusion-flow-study \
  --dataset swiss_roll \
  --train-steps 200 \
  --sample-count 500 \
  --sample-steps 50 \
  --ddpm-steps 50
```

Run the MNIST baseline. The first run downloads the official IDX files to
`data/mnist/`:

```bash
uv run diffusion-flow-study \
  --dataset mnist \
  --metric-sample-count 512
```

The run creates the following outputs / 实验将生成以下文件：

```text
artifacts/gaussian_mixture/
├── metrics.json     # configuration, losses, quality, steps, and timing
├── metrics.png      # side-by-side metric comparison
└── samples.png      # target and generated distributions
```

Useful options / 常用选项：

```bash
uv run diffusion-flow-study --help
uv run diffusion-flow-study --device cuda
uv run diffusion-flow-study --seed 123 --output-dir artifacts/seed_123
```

## Experimental contract / 实验控制条件

The MVP deliberately controls the following factors:

MVP 对以下因素进行显式控制：

| Factor / 因素 | Shared setting / 统一设置 |
|---|---|
| Data / 数据 | Eight equally weighted isotropic Gaussian modes on a ring / 环形分布的八个等权各向同性高斯模态 |
| Network / 网络 | Three-layer SiLU MLP with fixed Fourier time features / 三层 SiLU MLP 与固定 Fourier 时间特征 |
| Optimizer / 优化器 | AdamW with gradient clipping / 带梯度裁剪的 AdamW |
| Training data / 训练数据 | Fresh samples from the same population distribution / 每步从同一总体分布重新采样 |
| Evaluation / 评估 | Same reference size and random seed policy / 相同参考样本量与随机种子策略 |

Sampling steps are reported rather than forced to be identical: DDPM follows
its discrete training schedule, while Score-SDE and Flow Matching expose an
independent numerical integration budget. This makes quality-versus-NFE curves
a first-class follow-up experiment.

实验报告采样步数，而不强制三种方法使用完全相同的步数：DDPM 遵循离散训练调度，
Score-SDE 和 Flow Matching 则使用独立的数值积分预算。因此，质量–NFE 曲线将作为
后续重点实验。

## Metrics / 评估指标

- **RBF MMD²** measures kernel mean-embedding discrepancy; bandwidth is selected
  by the pooled median heuristic.
- **Sliced Wasserstein-2** averages one-dimensional transport discrepancies over
  random projections.
- **Runtime and steps** expose computational trade-offs that distributional
  metrics alone hide.
- **Stability** currently includes deterministic seeding, gradient clipping, and
  non-finite-loss failure checks. Multi-seed mean, standard deviation, and
  failure-rate reporting are the next milestone.

- **RBF MMD²** 衡量核均值嵌入差异，带宽通过合并样本的中位数启发式确定。
- **Sliced Wasserstein-2** 对随机投影上的一维传输差异取平均。
- **运行时间与步数** 揭示仅靠分布距离无法体现的计算权衡。
- **稳定性** 当前包含确定性种子、梯度裁剪与非有限 loss 检查；下一阶段将统计多随机种子
  的均值、标准差和失败率。

## Repository layout / 仓库结构

```text
src/diffusion_flow_study/
├── cli.py            # command-line entry point / 命令行入口
├── config.py         # experiment configuration / 实验配置
├── data.py           # synthetic and MNIST target distributions / 合成与 MNIST 目标分布
├── experiment.py     # training, evaluation, JSON, and figures / 训练、评估与制图
├── methods.py        # DDPM, Score-SDE, and Flow Matching / 三类生成方法
├── metrics.py        # MMD and sliced Wasserstein / 统计指标
└── models.py         # shared time-conditioned MLP / 共享时间条件网络
report/
└── technical_note.md # 6–10 page bilingual note outline / 双语报告大纲
tests/
└── test_smoke.py     # fast smoke tests / 快速冒烟测试
```

## Roadmap / 路线图

- [x] Two-dimensional Gaussian-mixture MVP.
- [ ] Multi-seed runs with confidence intervals and failure rates.
- [ ] Quality-versus-sampling-step Pareto curves.
- [x] Swiss-roll dataset path; intrinsic-dimension diagnostics remain open.
- [ ] Linear, cosine, and learned/noise-schedule ablations.
- [x] MNIST flattened-vector data path and baseline results.
- [ ] MNIST convolutional backbone with FID/KID-style evaluation.
- [ ] Simple reward-guided sampling and reward–fidelity trade-off curves.
- [ ] Final 6–10 page technical note with frozen configurations and results.

- [x] 二维高斯混合 MVP。
- [ ] 多随机种子实验、置信区间与失败率。
- [ ] 质量–采样步数 Pareto 曲线。
- [x] Swiss-roll 数据路径；内在维度诊断仍待补充。
- [ ] 线性、余弦及可学习噪声调度消融。
- [x] MNIST 展平向量数据路径与基线结果。
- [ ] MNIST 卷积主干与 FID/KID 风格评估。
- [ ] 简单 reward-guided sampling 与 reward–fidelity 权衡曲线。
- [ ] 使用冻结配置和最终结果完成 6–10 页 technical note。

## Reproducibility notes / 可复现性说明

Timing results depend on hardware and should be compared only within a single
machine/runtime configuration. The current code seeds Python, NumPy, PyTorch,
training data, and sampling streams, but exact GPU determinism is not claimed.
Commit the generated `uv.lock` file whenever dependencies change.

计时结果依赖具体硬件，只应在相同机器与运行环境内比较。当前代码分别为 Python、NumPy、
PyTorch、训练数据和采样流设置随机种子，但不保证 GPU 上的逐位确定性。依赖发生变化时，
应同时提交更新后的 `uv.lock`。

## License / 许可证

MIT
