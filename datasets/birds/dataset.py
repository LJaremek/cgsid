from __future__ import annotations

from cgsid.core.dataset import BaseSyntheticDataset
from cgsid.core.distribution import Relation, generate_birds_rows
from cgsid.core.labels import LabelSpec
from cgsid.enums import BeakState, BirdColor, BirdPosition, TimeOfDayBirds

from .clip_labels import build_birds_label_specs
from .correlations import build_birds_correlations


class BirdsDataset(BaseSyntheticDataset):
    name = "birds"
    clip_description = "Querying bird images with CLIP"

    def build_correlations(self) -> list[Relation]:
        return build_birds_correlations()

    def build_distribution_rows(self, *, size: int) -> list[dict[str, str]]:
        return generate_birds_rows(
            relations=self.build_correlations(),
            size=size,
            attributes=[
                BirdColor,
                BirdPosition,
                TimeOfDayBirds,
                BeakState,
            ],
        )

    def build_label_specs(self) -> list[LabelSpec]:
        return build_birds_label_specs()
