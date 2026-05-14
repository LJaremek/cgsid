from __future__ import annotations

import random

from cgsid.core.dataset import BaseSyntheticDataset
from cgsid.core.distribution import Relation, generate_domestic_rows
from cgsid.core.labels import LabelSpec
from cgsid.enums import (
    BallColor,
    BallObject,
    CameraAngle,
    CameraDistance,
    PetAnimal,
    PetColor,
    PetPosition,
    PetRoom,
    TimeOfDay,
)

from .clip_labels import build_domestic_label_specs
from .correlations import build_domestic_correlations


class DomesticDataset(BaseSyntheticDataset):
    name = "domestic"
    clip_description = "Querying domestic images with CLIP"

    def build_correlations(self) -> list[Relation]:
        return build_domestic_correlations()

    def build_distribution_rows(self, *, size: int) -> list[dict[str, str]]:
        return generate_domestic_rows(
            relations=self.build_correlations(),
            size=size,
            attributes=[
                PetAnimal,
                PetColor,
                PetPosition,
                PetRoom,
                TimeOfDay,
                BallObject,
                BallColor,
            ],
        )

    def build_label_specs(self) -> list[LabelSpec]:
        return build_domestic_label_specs()

    def generate_images(self, **kwargs):
        kwargs.setdefault("camera_angle", random.choice(list(CameraAngle)))
        kwargs.setdefault("camera_distance", random.choice(list(CameraDistance)))
        return super().generate_images(**kwargs)
