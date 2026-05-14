from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from cgsid.core.config import GenerationConfig
from datasets.domestic import DomesticDataset


def build_parser(dataset: DomesticDataset) -> argparse.ArgumentParser:
    config = GenerationConfig()
    parser = argparse.ArgumentParser(description="Generate the correlated domestic dataset.")
    parser.add_argument("--size", type=int, default=3_000)
    parser.add_argument("--distribution-csv", type=Path, default=dataset.default_distribution_csv)
    parser.add_argument("--output-dir", type=Path, default=dataset.default_output_dir)
    parser.add_argument("--model-name", default=config.model_name)
    parser.add_argument("--cache-dir", default=config.cache_dir)
    parser.add_argument("--width", type=int, default=config.width)
    parser.add_argument("--height", type=int, default=config.height)
    parser.add_argument("--num-inference-steps", type=int, default=config.num_inference_steps)
    parser.add_argument("--true-cfg-scale", type=float, default=config.true_cfg_scale)
    parser.add_argument("--skip-distribution", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    return parser


def main() -> None:
    dataset = DomesticDataset()
    args = build_parser(dataset).parse_args()
    distribution_csv = args.distribution_csv

    if not args.skip_distribution:
        distribution_csv = dataset.generate_distribution(output_path=args.distribution_csv, size=args.size)
        print(f"Saved domestic distribution to: {distribution_csv}")

    if args.skip_images:
        return

    if not distribution_csv.exists():
        raise FileNotFoundError(f"Distribution CSV not found: {distribution_csv}")

    metadata_path = dataset.generate_images(
        distribution_csv=distribution_csv,
        output_dir=args.output_dir,
        config=GenerationConfig(
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
        ),
    )
    print(f"Saved generated domestic metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
