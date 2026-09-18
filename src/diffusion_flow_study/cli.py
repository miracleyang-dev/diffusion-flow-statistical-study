"""Command-line interface."""

import argparse
import json

from .config import ExperimentConfig
from .experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("gaussian_mixture", "swiss_roll", "mnist"),
        default="gaussian_mixture",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sample-count", type=int, default=2_000)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--ddpm-steps", type=int, default=200)
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--metric-sample-count", type=int)
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        dataset=args.dataset,
        seed=args.seed,
        device=args.device,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
        sample_steps=args.sample_steps,
        ddpm_train_steps=args.ddpm_steps,
        output_dir=args.output_dir or f"artifacts/{args.dataset}",
        data_dir=args.data_dir,
        metric_sample_count=args.metric_sample_count,
    )
    print(json.dumps(run_experiment(config), indent=2))


if __name__ == "__main__":
    main()
