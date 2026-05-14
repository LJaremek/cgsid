from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from cgsid.core.config import ClipQueryConfig
from datasets.wild import WildDataset


def build_parser(dataset: WildDataset) -> argparse.ArgumentParser:
    defaults = ClipQueryConfig(dataset_dir=dataset.default_output_dir)
    parser = argparse.ArgumentParser(description="Query generated wild images with CLIP.")
    parser.add_argument("--dataset-dir", type=Path, default=defaults.dataset_dir)
    parser.add_argument("--output-dirname", default=defaults.output_dirname)
    parser.add_argument("--output-filename", default=defaults.output_filename)
    parser.add_argument("--label-catalog-filename", default=defaults.label_catalog_filename)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    return parser


def main() -> None:
    dataset = WildDataset()
    args = build_parser(dataset).parse_args()
    dataset.query_clip(
        ClipQueryConfig(
            dataset_dir=args.dataset_dir,
            output_dirname=args.output_dirname,
            output_filename=args.output_filename,
            label_catalog_filename=args.label_catalog_filename,
            device=args.device,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
