"""Synthetic and downloaded benchmark datasets."""

import gzip
import math
import struct
import urllib.request
from pathlib import Path

import torch
from torch import Tensor

MNIST_IMAGE_SHAPE = (28, 28)
MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
}
_MNIST_CACHE: dict[str, Tensor] = {}


def sample_gaussian_mixture(
    n: int,
    *,
    modes: int = 8,
    radius: float = 4.0,
    std: float = 0.25,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample an equally weighted ring of isotropic Gaussian components."""
    component = torch.randint(modes, (n,), device=device, generator=generator)
    angles = 2.0 * math.pi * component.float() / modes
    centers = radius * torch.stack((angles.cos(), angles.sin()), dim=1)
    noise = torch.randn(n, 2, device=device, generator=generator)
    return centers + std * noise


def sample_swiss_roll(
    n: int,
    *,
    turns: float = 1.5,
    radius_min: float = 1.5,
    radius_max: float = 4.5,
    height: float = 4.0,
    noise: float = 0.08,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample a noisy three-dimensional Swiss-roll manifold."""
    if turns <= 0:
        raise ValueError("turns must be positive")
    if radius_min <= 0 or radius_max <= radius_min:
        raise ValueError("radius_max must be greater than radius_min > 0")
    if height <= 0:
        raise ValueError("height must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    start = 1.5 * math.pi
    end = start + 2.0 * math.pi * turns
    angle = start + (end - start) * torch.rand(n, device=device, generator=generator)
    radius = radius_min + (radius_max - radius_min) * (angle - start) / (end - start)
    vertical = height * (torch.rand(n, device=device, generator=generator) - 0.5)
    points = torch.stack((radius * angle.cos(), vertical, radius * angle.sin()), dim=1)
    if noise:
        points = points + noise * torch.randn(points.shape, device=device, generator=generator)
    return points


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download MNIST from {url}") from error


def _read_mnist_images(path: Path) -> Tensor:
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    magic, count, rows, columns = struct.unpack(">IIII", raw[:16])
    if magic != 2051 or (rows, columns) != MNIST_IMAGE_SHAPE:
        raise ValueError(f"invalid MNIST image file: {path}")
    pixels = torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8)
    return pixels.reshape(count, rows * columns).float().div(255.0)


def _read_mnist_labels(path: Path) -> Tensor:
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"invalid MNIST label file: {path}")
    labels = torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8)
    return labels[:count].clone()


def _load_mnist(data_dir: str | Path = "data/mnist") -> Tensor:
    """Load and cache the official MNIST training images."""
    root = Path(data_dir)
    cache_key = str(root.resolve())
    if cache_key in _MNIST_CACHE:
        return _MNIST_CACHE[cache_key]

    image_path = root / "train-images-idx3-ubyte.gz"
    label_path = root / "train-labels-idx1-ubyte.gz"
    _download_if_missing(image_path, MNIST_URLS["train_images"])
    _download_if_missing(label_path, MNIST_URLS["train_labels"])
    images = _read_mnist_images(image_path)
    labels = _read_mnist_labels(label_path)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("MNIST image and label counts do not match")
    _MNIST_CACHE[cache_key] = images
    return images


def sample_mnist(
    n: int,
    *,
    data_dir: str | Path = "data/mnist",
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample flattened MNIST training images in the range [0, 1]."""
    images = _load_mnist(data_dir)
    indices = torch.randint(images.shape[0], (n,), device=device, generator=generator).cpu()
    return images.index_select(0, indices).to(device)


def dataset_dimension(name: str) -> int:
    """Return the ambient dimension for a supported dataset."""
    dimensions = {"gaussian_mixture": 2, "swiss_roll": 3, "mnist": 28 * 28}
    try:
        return dimensions[name]
    except KeyError as error:
        supported = ", ".join(dimensions)
        raise ValueError(f"unknown dataset {name!r}; expected one of: {supported}") from error


def sample_dataset(
    name: str,
    n: int,
    *,
    data_dir: str | Path = "data/mnist",
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one of the benchmark datasets using its default parameters."""
    if name == "gaussian_mixture":
        return sample_gaussian_mixture(n, device=device, generator=generator)
    if name == "swiss_roll":
        return sample_swiss_roll(n, device=device, generator=generator)
    if name == "mnist":
        return sample_mnist(n, data_dir=data_dir, device=device, generator=generator)
    dataset_dimension(name)
    raise AssertionError("unreachable")
