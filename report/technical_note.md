# DDPM、Score-SDE 与 Flow Matching：统计比较
# DDPM, Score-SDE, and Flow Matching: A Statistical Comparison

> First single-seed result snapshot, not the final multi-seed statistical
> comparison. The numerical values below are reproducibility anchors for the
> current implementation.
>
> 当前为单随机种子结果快照，不是最终多随机种子统计比较。下列数值用于
> 固定当前实现的可复现实验基线。

## Abstract / 摘要

We compare discrete denoising diffusion, continuous-time score-based diffusion,
and conditional flow matching under controlled data, architecture, optimization,
and evaluation settings. The study emphasizes distributional error,
quality–compute trade-offs, and run-to-run stability.

本文在统一的数据、网络结构、优化和评估设置下，对离散去噪扩散、连续时间 score-based
diffusion 与 conditional flow matching 进行比较，重点分析分布误差、质量–计算量权衡以及
不同运行之间的稳定性。

## 1. Research questions / 研究问题

1. Which objective best recovers separated modes under a matched training budget?
2. How rapidly does each sampler improve as the number of function evaluations grows?
3. How do noise schedules interact with low-dimensional or curved support?
4. What fidelity cost is induced by simple reward guidance?

1. 在匹配训练预算下，哪种目标函数能够最好地恢复相互分离的模态？
2. 随着函数评估次数增加，各采样器的质量提升速度如何？
3. 噪声调度如何与低维或弯曲支撑结构相互作用？
4. 简单 reward guidance 会带来多少保真度损失？

## 2. Methods / 方法

### 2.1 DDPM

Define the discrete forward process, epsilon-prediction objective, linear/cosine
schedules, and ancestral reverse transition.

定义离散前向过程、epsilon-prediction 目标、线性/余弦调度以及祖先反向转移。

### 2.2 Score-SDE

Define the variance-preserving SDE, marginal perturbation kernel, weighted
denoising score-matching objective, and Euler–Maruyama reverse solver.

定义 variance-preserving SDE、边缘扰动核、加权 denoising score-matching 目标和
Euler–Maruyama 反向求解器。

### 2.3 Flow Matching

Define the Gaussian-to-data probability path, conditional velocity target, and
Euler integration of the learned ODE.

定义高斯到数据的概率路径、条件速度目标以及学习所得 ODE 的 Euler 积分。

## 3. Experimental design / 实验设计

- Data: eight-mode Gaussian mixture, Swiss roll, and MNIST.
- Controls: common capacity, optimizer family, batch size, update budget, and seeds.
- Primary metrics: RBF MMD² and sliced Wasserstein-2. MNIST metrics are
  computed in pixel space on a 512-sample evaluation subset; FID/KID is not
  implemented in this baseline.
- Compute metrics: sampling steps/NFE and wall-clock time.
- Stability: mean ± standard deviation, bootstrap confidence intervals, and failure rate.
- Statistical protocol: at least five seeds; paired seed-level comparisons where possible.

- 数据：八模态 Gaussian mixture、Swiss roll 与 MNIST。
- 控制变量：统一模型容量、优化器类型、batch size、更新预算和随机种子集合。
- 主要指标：RBF MMD² 与 Sliced Wasserstein-2；MNIST 当前在像素空间上使用
  512 个评估样本，FID/KID 尚未实现。
- 计算指标：采样步数/NFE 与 wall-clock time。
- 稳定性：均值 ± 标准差、bootstrap 置信区间及失败率。
- 统计方案：至少五个随机种子；条件允许时采用随机种子级别的配对比较。

## 4. Results / 结果

### 4.0 Result snapshot

All three formal runs used seed 42, 5,000 training updates, batch size 512,
hidden width 128, and 2,000 generated/reference samples. DDPM used 200
sampling steps; Score-SDE and Flow Matching used 100 steps. MNIST metrics use
the first 512 reference/generated samples because pairwise pixel-space
distances are substantially more expensive in 784 dimensions.

| Dataset | Method | RBF MMD² | Sliced W-2 | Train (s) | Sample (s) |
|---|---|---:|---:|---:|---:|
| Gaussian mixture | DDPM | 0.000894 | 0.2739 | 31.36 | 0.327 |
| Gaussian mixture | Score-SDE | 0.004960 | 0.4774 | 19.13 | 0.133 |
| Gaussian mixture | Flow Matching | 0.000670 | 0.2547 | 16.52 | 0.136 |
| Swiss roll | DDPM | 0.003029 | 0.1900 | 30.56 | 0.564 |
| Swiss roll | Score-SDE | 0.001890 | 0.1606 | 30.58 | 0.256 |
| Swiss roll | Flow Matching | 0.000540 | 0.0877 | 29.29 | 0.242 |
| MNIST | DDPM | 0.393545 | 3.2233 | 94.36 | 6.797 |
| MNIST | Score-SDE | 0.401156 | 175.5247 | 96.66 | 3.958 |
| MNIST | Flow Matching | 0.292336 | 0.6883 | 103.74 | 1.006 |

The values are not directly comparable across datasets because ambient
dimension, support geometry, and the MNIST metric sample count differ.

### 4.1 Gaussian mixture

The Gaussian-mixture baseline is recovered most closely by Flow Matching and
DDPM. Flow Matching has the lowest MMD² and sliced Wasserstein-2 in this run,
while DDPM is competitive at the cost of 200 reverse steps. The generated
sample and metric figures are stored in `artifacts/gaussian_mixture/`.

本次运行中 Flow Matching 的 MMD² 与 Sliced Wasserstein-2 最低，DDPM
紧随其后但使用了更多反向采样步。样本图和指标图位于
`artifacts/gaussian_mixture/`。

### 4.2 Swiss roll and low-dimensional structure

Flow Matching gives the best ambient-space metrics on the Swiss roll, followed
by Score-SDE. The 3D target/generated panels are readable and are stored in
`artifacts/swiss_roll/`. Distance-to-manifold and neighborhood-preservation
diagnostics are not yet implemented, so this section currently reports only
ambient distributional metrics.

本次运行中 Flow Matching 的环境空间指标最好，Score-SDE 次之。三维
目标/生成样本图位于 `artifacts/swiss_roll/`。到流形距离和局部邻域保持
诊断尚未实现，因此本节目前只报告环境空间分布指标。

### 4.3 MNIST

MNIST is connected through the official IDX files and a flattened 784-dimensional
representation. Flow Matching has the lowest pixel-space MMD² and sliced
Wasserstein-2, but the representative sample grid in `artifacts/mnist/`
contains mostly noise rather than recognizable digits. This is a failed image
generation baseline, not evidence of good MNIST fidelity. The current result
should be replaced by a convolutional backbone and FID/KID evaluation before
making image-quality claims.

MNIST 通过官方 IDX 文件接入，并使用展平的 784 维表示。Flow Matching
的像素空间 MMD² 与 Sliced Wasserstein-2 最低，但
`artifacts/mnist/` 中的代表性样本主要仍是噪声，不能视为成功的数字生成。
后续需要卷积主干和 FID/KID 评估。

## 5. Ablations and guided sampling / 消融与引导采样

Compare linear and cosine schedules. Add a differentiable reward favoring a
chosen mixture mode or digit class, then report reward improvement against MMD,
Wasserstein/FID, and mode coverage.

比较线性与余弦噪声调度。加入偏向指定混合模态或数字类别的可微 reward，并对照 MMD、
Wasserstein/FID 和模态覆盖率，报告 reward 提升带来的分布保真度代价。

## 6. Discussion and limitations / 讨论与局限

Separate optimization effects from numerical-solver effects; discuss metric
sensitivity, small-model limitations, timing comparability, and the gap between
toy geometry and image generation.

The present numbers are single-seed exploratory results, not confidence
intervals or failure-rate estimates. The Gaussian mixture and Swiss roll runs
use 2,000 samples for both metrics, while MNIST uses 512. MNIST is currently
trained with the same flattened-vector MLP used for the low-dimensional
benchmarks; the sample grid shows that this capacity and parameterization are
insufficient for recognizable digits. FID/KID, class coverage, manifold
diagnostics, and multi-seed uncertainty remain open.

区分优化过程与数值求解器造成的影响，并讨论指标敏感性、小模型局限、计时可比性，以及
二维玩具几何与真实图像生成之间的差距。

当前结果是单随机种子的探索性结果，不包含置信区间或失败率。Gaussian
mixture 和 Swiss roll 的指标均使用 2,000 个样本，MNIST 使用 512 个样本。
MNIST 仍使用与低维实验相同的展平向量 MLP；样本图表明该容量和参数化
不足以生成可辨识数字。FID/KID、类别覆盖率、流形诊断和多随机种子不确定性
仍待补充。

## 7. Conclusion / 结论

Across the two low-dimensional benchmarks, Flow Matching is the strongest
single-seed baseline by the reported distributional metrics. MNIST exposes the
limitation of transferring the same small vector MLP to image generation:
numeric pixel-space metrics alone do not establish visual fidelity. The next
technically necessary step is a convolutional MNIST model followed by
image-specific evaluation.

在两个低维基准上，Flow Matching 在本次单随机种子分布指标中表现最好。
MNIST 则暴露出将小型向量 MLP 直接迁移到图像生成的局限：像素空间数值
指标不能单独证明视觉保真度。下一步应使用卷积 MNIST 模型并加入图像专用评估。

## Reproducibility appendix / 可复现性附录

Record commit hash, dependency lockfile, hardware, commands, seeds, exact
configurations, and per-seed raw results.

Current artifact roots:

- `artifacts/gaussian_mixture/metrics.json`
- `artifacts/swiss_roll/metrics.json`
- `artifacts/mnist/metrics.json`

MNIST data is cached under `data/mnist/`. The three runs used:

```text
python -m diffusion_flow_study.cli --dataset gaussian_mixture
python -m diffusion_flow_study.cli --dataset swiss_roll
python -m diffusion_flow_study.cli --dataset mnist --metric-sample-count 512
```

记录 commit hash、依赖锁文件、硬件、运行命令、随机种子、完整配置和各随机种子的原始结果。
