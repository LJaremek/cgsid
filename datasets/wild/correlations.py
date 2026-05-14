from __future__ import annotations

from cgsid.core.distribution import Relation
from cgsid.enums import TimeOfDay, WildPredator, WildPredatorColor, WildSeason


def build_wild_correlations() -> list[Relation]:
    return [
        Relation(
            WildPredator.FOX,
            {
                WildSeason.SUMMER: 0.85,
                WildSeason.WINTER: 0.15,
            },
        ),
        Relation(
            WildPredator.DOG,
            {
                WildSeason.SUMMER: 0.20,
                WildSeason.WINTER: 0.80,
            },
        ),
        Relation(
            WildPredator.CAT,
            {
                WildSeason.SUMMER: 0.65,
                WildSeason.WINTER: 0.35,
            },
        ),
        Relation(
            TimeOfDay.DAY,
            {
                WildPredatorColor.RED: 0.45,
                WildPredatorColor.WHITE: 0.30,
                WildPredatorColor.BLACK: 0.15,
                WildPredatorColor.BROWN: 0.10,
            },
        ),
        Relation(
            TimeOfDay.EVENING,
            {
                WildPredatorColor.RED: 0.10,
                WildPredatorColor.WHITE: 0.15,
                WildPredatorColor.BLACK: 0.35,
                WildPredatorColor.BROWN: 0.40,
            },
        ),
    ]
