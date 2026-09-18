"""Minimal, comparable DDPM, VP Score-SDE, and Flow Matching methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import Tensor, nn

BatchSampler = Callable[[int], Tensor]


class GenerativeMethod(ABC):
    """Common training interface for the controlled benchmark."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        data_dim: int = 2,
    ) -> None:
        if data_dim <= 0:
            raise ValueError("data_dim must be positive")
        self.model = model.to(device)
        self.device = torch.device(device)
        self.data_dim = data_dim

    @abstractmethod
    def loss(self, clean: Tensor) -> Tensor:
        """Return the method-specific training objective."""

    @abstractmethod
    @torch.no_grad()
    def sample(self, n: int, steps: int, generator: torch.Generator) -> Tensor:
        """Generate samples."""

    def fit(
        self,
        data_sampler: BatchSampler,
        *,
        train_steps: int,
        batch_size: int,
        learning_rate: float,
        log_every: int = 500,
    ) -> list[float]:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        history: list[float] = []
        self.model.train()
        for step in range(1, train_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            value = self.loss(data_sampler(batch_size))
            if not torch.isfinite(value):
                raise FloatingPointError(f"non-finite loss at step {step}")
            value.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            optimizer.step()
            if step == 1 or step % log_every == 0 or step == train_steps:
                history.append(float(value.detach()))
        return history


class DDPM(GenerativeMethod):
    """Discrete variance-preserving diffusion with epsilon prediction."""

    def __init__(
        self,
        model: nn.Module,
        *,
        diffusion_steps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str | torch.device = "cpu",
        data_dim: int = 2,
    ) -> None:
        super().__init__(model, device, data_dim)
        self.diffusion_steps = diffusion_steps
        self.betas = torch.linspace(beta_start, beta_end, diffusion_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def loss(self, clean: Tensor) -> Tensor:
        n = clean.shape[0]
        index = torch.randint(self.diffusion_steps, (n,), device=self.device)
        noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bars[index, None]
        noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
        predicted = self.model(noisy, index.float() / (self.diffusion_steps - 1))
        return (predicted - noise).square().mean()

    @torch.no_grad()
    def sample(self, n: int, steps: int, generator: torch.Generator) -> Tensor:
        if steps != self.diffusion_steps:
            raise ValueError(f"DDPM requires steps={self.diffusion_steps}; got {steps}")
        self.model.eval()
        x = torch.randn(n, self.data_dim, device=self.device, generator=generator)
        for index in reversed(range(self.diffusion_steps)):
            t = torch.full((n,), index / (self.diffusion_steps - 1), device=self.device)
            predicted_noise = self.model(x, t)
            beta, alpha = self.betas[index], self.alphas[index]
            alpha_bar = self.alpha_bars[index]
            mean = (x - beta * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
            if index > 0:
                previous_bar = self.alpha_bars[index - 1]
                variance = beta * (1.0 - previous_bar) / (1.0 - alpha_bar)
                noise = torch.randn(x.shape, device=self.device, generator=generator)
                x = mean + variance.sqrt() * noise
            else:
                x = mean
        return x


class VPScoreSDE(GenerativeMethod):
    """Continuous-time VP-SDE trained with denoising score matching."""

    def __init__(
        self,
        model: nn.Module,
        *,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        device: str | torch.device = "cpu",
        data_dim: int = 2,
    ) -> None:
        super().__init__(model, device, data_dim)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _marginal(self, t: Tensor) -> tuple[Tensor, Tensor]:
        log_mean = -0.25 * (self.beta_max - self.beta_min) * t.square() - 0.5 * self.beta_min * t
        mean_scale = log_mean.exp()
        std = (1.0 - mean_scale.square()).clamp_min(1e-6).sqrt()
        return mean_scale, std

    def loss(self, clean: Tensor) -> Tensor:
        n = clean.shape[0]
        t = torch.rand(n, device=self.device) * 0.999 + 0.001
        mean_scale, std = self._marginal(t)
        noise = torch.randn_like(clean)
        noisy = mean_scale[:, None] * clean + std[:, None] * noise
        score = self.model(noisy, t)
        return (std[:, None] * score + noise).square().mean()

    @torch.no_grad()
    def sample(self, n: int, steps: int, generator: torch.Generator) -> Tensor:
        self.model.eval()
        x = torch.randn(n, self.data_dim, device=self.device, generator=generator)
        dt = (1.0 - 1e-3) / steps
        for index in range(steps):
            time = 1.0 - index * dt
            t = torch.full((n,), time, device=self.device)
            beta = self.beta_min + time * (self.beta_max - self.beta_min)
            reverse_drift = 0.5 * beta * x + beta * self.model(x, t)
            noise = torch.randn(x.shape, device=self.device, generator=generator)
            x = x + reverse_drift * dt + (beta * dt) ** 0.5 * noise
        return x


class FlowMatching(GenerativeMethod):
    """Conditional flow matching on straight Gaussian-to-data paths."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        data_dim: int = 2,
    ) -> None:
        super().__init__(model, device, data_dim)

    def loss(self, clean: Tensor) -> Tensor:
        n = clean.shape[0]
        source = torch.randn_like(clean)
        t = torch.rand(n, device=self.device)
        interpolant = (1.0 - t[:, None]) * source + t[:, None] * clean
        target_velocity = clean - source
        return (self.model(interpolant, t) - target_velocity).square().mean()

    @torch.no_grad()
    def sample(self, n: int, steps: int, generator: torch.Generator) -> Tensor:
        self.model.eval()
        x = torch.randn(n, self.data_dim, device=self.device, generator=generator)
        dt = 1.0 / steps
        for index in range(steps):
            t = torch.full((n,), (index + 0.5) * dt, device=self.device)
            x = x + dt * self.model(x, t)
        return x
