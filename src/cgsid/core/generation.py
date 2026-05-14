from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from cgsid.enums import (
    BallColor,
    BallObject,
    BeakState,
    BirdColor,
    BirdPosition,
    CameraAngle,
    CameraDistance,
    PetAnimal,
    PetColor,
    PetPosition,
    PetRoom,
    TimeOfDay,
    WildPredator,
    WildPredatorColor,
    WildPrey,
    WildPreyColor,
    WildSeason,
)
from cgsid.core.distribution import DATASET_COLUMNS, generate_birds_rows
from cgsid.prompts import build_flux_bird_prompt, build_flux_pet_prompt, build_wild_animal_prompt


def default_birds_relations():
    from cgsid.core.distribution import Relation

    return [
        Relation(
            TimeOfDay.DAY,
            {
                BirdPosition.FLYING: 0.60,
                BirdPosition.NEST: 0.10,
                BirdPosition.BRANCH: 0.30,
            },
        ),
        Relation(
            BirdColor.BLUE,
            {
                BirdPosition.FLYING: 0.50,
                BirdPosition.NEST: 0.15,
                BirdPosition.BRANCH: 0.35,
            },
        ),
        Relation(
            BirdPosition.FLYING,
            {
                BeakState.EMPTY: 0.85,
                BeakState.WORM: 0.15,
            },
        ),
    ]


def generate_birds_distribution_csv(
    *,
    output_path: str | Path,
    birds_size: int,
) -> Path:
    distribution_path = Path(output_path)
    distribution_path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_birds_rows(
        relations=default_birds_relations(),
        size=birds_size,
    )
    write_metadata_rows(distribution_path, rows, append=False)
    return distribution_path


def load_distribution_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [normalize_row(row) for row in reader]


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {column: str(row.get(column, "none")).strip() or "none" for column in DATASET_COLUMNS}


def build_prompt_from_row(
    row: dict[str, str],
    *,
    wild_camera_angle: CameraAngle,
    wild_camera_distance: CameraDistance,
) -> str:
    subset = row["subset"]

    if subset == "wild":
        return build_wild_animal_prompt(
            chasing_animal=WildPredator(row["object_1"]),
            chased_animal=WildPrey(row["object_2"]),
            time_of_day=TimeOfDay(row["day_time"]),
            season=WildSeason(row["season"]),
            chasing_animal_color=WildPredatorColor(row["object_1_color"]),
            chased_animal_color=WildPreyColor(row["object_2_color"]),
            camera_angle=wild_camera_angle,
            camera_distance=wild_camera_distance,
        )

    if subset == "domestic":
        is_playing_with_ball = row["interaction"] == "playing_with_ball" and row["object_2"] != "none"
        room = PetRoom.LIVING_ROOM if row["environment"] == "living_room" else PetRoom.KITCHEN
        ball_color = None
        if is_playing_with_ball and row["object_2_color"] != "none":
            ball_color = BallColor(row["object_2_color"])

        return build_flux_pet_prompt(
            animal=PetAnimal(row["object_1"]),
            animal_color=PetColor(row["object_1_color"]),
            position=PetPosition(row["pose"]),
            is_playing_with_ball=is_playing_with_ball,
            room=room,
            time_of_day=TimeOfDay(row["day_time"]),
            ball_object=BallObject.BALL if is_playing_with_ball else None,
            ball_color=ball_color,
        )

    if subset == "birds":
        return build_flux_bird_prompt(
            bird_color=BirdColor(row["object_1_color"]),
            position=bird_position_from_row(row),
            time_of_day=TimeOfDay(row["day_time"]),
            beak_state=BeakState.WORM if row["interaction"] == "holding_worm_in_beak" else BeakState.EMPTY,
        )

    raise ValueError(f"Unsupported subset: {subset!r}")


def bird_position_from_row(row: dict[str, str]) -> BirdPosition:
    pose = row["pose"]
    if pose == "flying":
        return BirdPosition.FLYING
    if pose == "sitting_in_nest":
        return BirdPosition.NEST
    if pose == "standing_on_branch":
        return BirdPosition.BRANCH
    raise ValueError(f"Unsupported bird pose in CSV: {pose!r}")


def load_existing_metadata(metadata_path: Path) -> tuple[int, int]:
    if not metadata_path.exists():
        return 0, 0

    processed_rows = 0
    next_index = 0
    with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            processed_rows += 1
            image_path = str(row.get("image_path", ""))
            match = re.match(r"sample_(\d{6})\.png$", image_path)
            if match:
                next_index = max(next_index, int(match.group(1)) + 1)

    return processed_rows, next_index


def write_metadata_rows(
    metadata_path: Path,
    rows: Iterable[dict[str, str]],
    *,
    append: bool,
) -> None:
    rows = list(rows)
    fieldnames = list(DATASET_COLUMNS)
    if any("prompt" in row for row in rows):
        fieldnames.append("prompt")

    mode = "a" if append and metadata_path.exists() else "w"
    with metadata_path.open(mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def generate_dataset_from_csv(
    *,
    distribution_csv: str | Path,
    output_dir: str | Path,
    config,
    wild_camera_angle: CameraAngle,
    wild_camera_distance: CameraDistance,
) -> Path:
    from tqdm import tqdm

    from cgsid.models import FluxModel

    rows = load_distribution_rows(distribution_csv)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = target_dir / "metadata.csv"

    processed_rows, next_index = load_existing_metadata(metadata_path)
    pending_rows = rows[processed_rows:]
    if not pending_rows:
        print("Dataset already generated for all rows from the distribution CSV.")
        return metadata_path

    model = FluxModel(model_name=config.model_name, cache_dir=config.cache_dir)
    generated_rows: list[dict[str, str]] = []

    progress = tqdm(pending_rows, desc="Generating dataset from CSV")
    for row in progress:
        prompt = build_prompt_from_row(
            row,
            wild_camera_angle=wild_camera_angle,
            wild_camera_distance=wild_camera_distance,
        )
        image = model.generate(
            prompt,
            width=config.width,
            height=config.height,
            num_inference_steps=config.num_inference_steps,
            true_cfg_scale=config.true_cfg_scale,
        )
        image_name = f"sample_{next_index:06d}.png"
        model.save(image, str(target_dir / image_name))

        output_row = dict(row)
        output_row["image_path"] = image_name
        output_row["prompt"] = prompt
        generated_rows.append(output_row)
        next_index += 1

        if len(generated_rows) >= 25:
            write_metadata_rows(metadata_path, generated_rows, append=True)
            generated_rows.clear()

    if generated_rows:
        write_metadata_rows(metadata_path, generated_rows, append=True)

    return metadata_path
