from __future__ import annotations

from cgsid.core.distribution import Relation
from cgsid.enums import BirdColor, BirdPosition, TimeOfDayBirds


def build_birds_correlations() -> list[Relation]:
    return [
        Relation(
            BirdPosition.FLYING,
            {
                BirdColor.BLUE: 0.30,
                BirdColor.RED: 0.08,
                BirdColor.YELLOW: 0.28,
                BirdColor.GREEN: 0.22,
                BirdColor.BLACK: 0.06,
                BirdColor.WHITE: 0.06,
            },
        ),
        Relation(
            BirdPosition.NEST,
            {
                BirdColor.BLUE: 0.06,
                BirdColor.RED: 0.30,
                BirdColor.YELLOW: 0.08,
                BirdColor.GREEN: 0.06,
                BirdColor.BLACK: 0.25,
                BirdColor.WHITE: 0.25,
            },
        ),
        Relation(
            BirdPosition.BRANCH,
            {
                BirdColor.BLUE: 0.08,
                BirdColor.RED: 0.12,
                BirdColor.YELLOW: 0.16,
                BirdColor.GREEN: 0.36,
                BirdColor.BLACK: 0.14,
                BirdColor.WHITE: 0.14,
            },
        ),
        Relation(
            BirdPosition.FLYING,
            {
                TimeOfDayBirds.DAY: 0.90,
                TimeOfDayBirds.EVENING: 0.10,
            },
        ),
        Relation(
            BirdPosition.NEST,
            {
                TimeOfDayBirds.DAY: 0.20,
                TimeOfDayBirds.EVENING: 0.80,
            },
        ),
        Relation(
            BirdPosition.BRANCH,
            {
                TimeOfDayBirds.DAY: 0.70,
                TimeOfDayBirds.EVENING: 0.30,
            },
        ),
    ]
