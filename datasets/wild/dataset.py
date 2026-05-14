from __future__ import annotations

from cgsid.core.dataset import BaseSyntheticDataset
from cgsid.core.distribution import Relation, generate_wild_rows
from cgsid.core.labels import LabelSpec
from cgsid.enums import TimeOfDay, WildPredator, WildPredatorColor, WildPreyColor, WildSeason

from .clip_labels import build_wild_label_specs
from .correlations import build_wild_correlations


class WildDataset(BaseSyntheticDataset):
    name = "wild"
    clip_description = "Querying wild images with CLIP"

    def build_correlations(self) -> list[Relation]:
        return build_wild_correlations()

    def build_distribution_rows(self, *, size: int) -> list[dict[str, str]]:
        rows = generate_wild_rows(
            relations=self.build_correlations(),
            size=size,
            attributes=[
                WildPredator,
                WildPredatorColor,
                WildPreyColor,
                TimeOfDay,
                WildSeason,
            ],
        )
        for row in rows:
            row["object_2"] = "hare"
        return rows

    def build_label_specs(self) -> list[LabelSpec]:
        return build_wild_label_specs()
