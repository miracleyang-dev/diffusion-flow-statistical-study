# DDPM、Score-SDE 与 Flow Matching：统计比较
# DDPM, Score-SDE, and Flow Matching: A Statistical Comparison

> Working outline for the final 6–10 page note. Numerical claims remain blank
> until the multi-seed benchmark is frozen.
>
> 最终 6–10 页技术报告的工作大纲。在多随机种子基准配置冻结前，不填入定量结论。

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
- Primary metrics: RBF MMD², sliced Wasserstein-2, and MNIST FID/KID.
- Compute metrics: sampling steps/NFE and wall-clock time.
- Stability: mean ± standard deviation, bootstrap confidence intervals, and failure rate.
- Statistical protocol: at least five seeds; paired seed-level comparisons where possible.

- 数据：八模态 Gaussian mixture、Swiss roll 与 MNIST。
- 控制变量：统一模型容量、优化器类型、batch size、更新预算和随机种子集合。
- 主要指标：RBF MMD²、Sliced Wasserstein-2，以及 MNIST 上的 FID/KID。
- 计算指标：采样步数/NFE 与 wall-clock time。
- 稳定性：均值 ± 标准差、bootstrap 置信区间及失败率。
- 统计方案：至少五个随机种子；条件允许时采用随机种子级别的配对比较。

## 4. Results / 结果

### 4.1 Gaussian mixture

Insert target/sample panels, metric table, and quality-versus-NFE curve.

插入目标/生成样本图、指标表格和质量–NFE 曲线。

### 4.2 Swiss roll and low-dimensional structure

Report ambient distributional metrics together with distance-to-manifold and
neighborhood-preservation diagnostics.

同时报告环境空间分布指标、到流形距离及局部邻域保持诊断。

### 4.3 MNIST

Report image metrics, class coverage, throughput, and representative samples.

报告图像指标、类别覆盖率、吞吐量和代表性生成样本。

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

区分优化过程与数值求解器造成的影响，并讨论指标敏感性、小模型局限、计时可比性，以及
二维玩具几何与真实图像生成之间的差距。

## 7. Conclusion / 结论

Summarize empirically supported trade-offs only after results are frozen.

仅在实验配置和结果冻结后，总结具有实证支持的权衡关系。

## Reproducibility appendix / 可复现性附录

Record commit hash, dependency lockfile, hardware, commands, seeds, exact
configurations, and per-seed raw results.

记录 commit hash、依赖锁文件、硬件、运行命令、随机种子、完整配置和各随机种子的原始结果。
