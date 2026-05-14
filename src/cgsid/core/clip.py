from __future__ import annotations

from dataclasses import replace
from typing import Iterable
from pathlib import Path
import csv

from .config import ClipQueryConfig
from .labels import LabelSpec


def list_image_paths(dataset_dir: Path) -> list[Path]:
    metadata_path = dataset_dir / "metadata.csv"
    if metadata_path.exists():
        return list_image_paths_from_metadata(dataset_dir, metadata_path)
    return sorted(dataset_dir.glob("*.png"))


def list_image_paths_from_metadata(dataset_dir: Path, metadata_path: Path) -> list[Path]:
    image_paths: list[Path] = []
    with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "image_path" not in (reader.fieldnames or []):
            raise KeyError(f"{metadata_path} must contain an 'image_path' column")

        for row in reader:
            image_name = str(row.get("image_path", "")).strip()
            if image_name:
                image_paths.append(dataset_dir / image_name)
    return image_paths


def save_label_catalog(path: Path, labels: list[LabelSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    include_subset = any(label.subset for label in labels)
    fieldnames = ["key", "text", "group", "prompt_count"]
    if include_subset:
        fieldnames.insert(2, "subset")

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for label in labels:
            row = {
                "key": label.key,
                "text": " | ".join(label.text_tuple),
                "group": label.group,
                "prompt_count": len(label.text_tuple),
            }
            if include_subset:
                row["subset"] = label.subset
            writer.writerow(row)


def query_dataset_with_clip(
    *,
    config: ClipQueryConfig,
    labels: list[LabelSpec],
    description: str,
) -> None:
    from PIL import Image
    from tqdm import tqdm

    from .clip_model import CLIPS, get_clip_with_processor, query_clip_with_processor

    image_paths = list_image_paths(config.dataset_dir)
    if not image_paths:
        raise FileNotFoundError(f"No .png images found in {config.dataset_dir}")

    missing_paths = [path for path in image_paths if not path.exists()]
    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths[:5])
        raise FileNotFoundError(f"Missing image files referenced by metadata.csv: {preview}")

    label_groups = build_label_groups(labels)
    save_label_catalog(config.label_catalog_csv, labels)

    model_config = replace(CLIPS.openai_clip, device=config.device)
    clip_model, processor = get_clip_with_processor(model_config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    with config.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["image_path"] + [label.key for label in labels]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for batch_paths in tqdm(
            batched(image_paths, config.batch_size),
            desc=description,
            total=batch_count(len(image_paths), config.batch_size),
        ):
            batch_images = load_images(batch_paths, image_loader=Image.open)
            probabilities_by_key = query_label_groups(
                images=batch_images,
                label_groups=label_groups,
                clip_model=clip_model,
                processor=processor,
                query_clip_with_processor=query_clip_with_processor,
            )
            write_batch_scores(
                writer=writer,
                batch_paths=batch_paths,
                labels=labels,
                probabilities_by_key=probabilities_by_key,
            )

    print("\nSaved CLIP outputs to:")
    print(f"  scores: {config.output_csv}")
    print(f"  labels: {config.label_catalog_csv}")


def build_label_groups(labels: list[LabelSpec]) -> dict[str, list[LabelSpec]]:
    groups: dict[str, list[LabelSpec]] = {}
    for label in labels:
        group_key = f"{label.subset}:{label.group}" if label.subset else label.group
        groups.setdefault(group_key, []).append(label)
    return groups


def query_label_groups(*, images, label_groups, clip_model, processor, query_clip_with_processor):
    import torch

    scores_by_key = {}
    for group_labels in label_groups.values():
        texts, label_text_indices = flatten_group_texts(group_labels)
        prompt_probabilities = query_clip_with_processor(images, texts, clip_model, processor)
        label_probabilities = aggregate_group_probabilities(prompt_probabilities, label_text_indices)
        normalized_label_probabilities = label_probabilities / label_probabilities.sum(dim=1, keepdim=True)

        for label, probabilities in zip(group_labels, normalized_label_probabilities.T, strict=True):
            scores_by_key[label.key] = probabilities

    return scores_by_key


def flatten_group_texts(labels: list[LabelSpec]) -> tuple[list[str], list[list[int]]]:
    texts: list[str] = []
    label_text_indices: list[list[int]] = []
    for label in labels:
        label_texts = label.text_tuple
        indices = list(range(len(texts), len(texts) + len(label_texts)))
        texts.extend(label_texts)
        label_text_indices.append(indices)
    return texts, label_text_indices


def aggregate_group_probabilities(probabilities, label_text_indices: list[list[int]]):
    import torch

    label_scores = [probabilities[:, indices].sum(dim=1) for indices in label_text_indices]
    return torch.stack(label_scores, dim=1)


def load_images(batch_paths: list[Path], image_loader) -> list:
    images = []
    for path in batch_paths:
        with image_loader(path) as image:
            images.append(image.convert("RGB"))
    return images


def write_batch_scores(*, writer, batch_paths: list[Path], labels: list[LabelSpec], probabilities_by_key) -> None:
    for image_index, image_path in enumerate(batch_paths):
        row = {"image_path": image_path.name}
        row.update(
            {
                label.key: str(float(probabilities_by_key[label.key][image_index]))
                for label in labels
            }
        )
        writer.writerow(row)


def batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    for start_index in range(0, len(items), batch_size):
        yield items[start_index:start_index + batch_size]


def batch_count(item_count: int, batch_size: int) -> int:
    return (item_count + batch_size - 1) // batch_size
