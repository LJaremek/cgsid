from __future__ import annotations

from cgsid.core.distribution import Relation
from cgsid.enums import BallObject, PetAnimal, PetRoom


def build_domestic_correlations() -> list[Relation]:
    return [
        Relation(
            PetAnimal.DOG,
            {
                PetRoom.LIVING_ROOM: 0.86,
                PetRoom.KITCHEN: 0.14,
            },
        ),
        Relation(
            PetAnimal.CAT,
            {
                PetRoom.LIVING_ROOM: 0.25,
                PetRoom.KITCHEN: 0.75,
            },
        ),
        Relation(
            PetRoom.LIVING_ROOM,
            {
                BallObject.BALL: 0.91,
                BallObject.NONE: 0.09,
            },
        ),
        Relation(
            PetRoom.KITCHEN,
            {
                BallObject.BALL: 0.17,
                BallObject.NONE: 0.83,
            },
        ),
        Relation(
            BallObject.BALL,
            {
                PetAnimal.DOG: 0.83,
                PetAnimal.CAT: 0.17,
            },
        ),
        Relation(
            BallObject.NONE,
            {
                PetAnimal.DOG: 0.20,
                PetAnimal.CAT: 0.80,
            },
        ),
    ]
