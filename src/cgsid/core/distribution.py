from itertools import product
from enum import Enum
import csv
from random import Random
from typing import Type

from cgsid.enums import BallColor, BallObject, BeakState, BirdColor, BirdPosition, PetAnimal, PetColor, PetPosition, PetRoom
from cgsid.enums.common import TimeOfDay
from cgsid.enums.wild_animals import WildPredator, WildPredatorColor
from cgsid.enums.wild_animals import WildPrey, WildPreyColor
from cgsid.enums import wild_animals

DATASET_COLUMNS = [
    "image_path",
    "subset",
    "object_1",
    "object_1_color",
    "object_2",
    "object_2_color",
    "interaction",
    "environment",
    "pose",
    "day_time",
    "season",
]

ATTRIBUTE_COLUMN_MAP: dict[type[Enum], str] = {
    WildPredator: "object_1",
    WildPredatorColor: "object_1_color",
    WildPrey: "object_2",
    WildPreyColor: "object_2_color",
    TimeOfDay: "day_time",
    wild_animals.WildSeason: "season",
}

BIRD_POSE_MAP = {
    BirdPosition.FLYING: "flying",
    BirdPosition.NEST: "sitting_in_nest",
    BirdPosition.BRANCH: "standing_on_branch",
}

BIRD_ENVIRONMENT_MAP = {
    BirdPosition.FLYING: "open_sky",
    BirdPosition.NEST: "tree_nest",
    BirdPosition.BRANCH: "tree_branch",
}


def _csv_value(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


class Relation:
    def __init__(
            self,
            parent: Enum,
            children: dict[Enum, float | None]
            ) -> None:
        self.parent = parent
        self.children = self._validate_children(children)

    def _validate_children(
            self,
            children: dict[Enum, float | None]
            ) -> dict[Enum, float]:
        total_prob = 0.0
        children_without_prob = 0
        children_types: Enum = None

        for child, prob in children.items():
            if children_types and child.__class__ != children_types:
                raise ValueError(f"All children must be of the same Enum type. Found: {child.__class__} and {children_types}")
            else:
                children_types = child.__class__

            if prob is not None:
                total_prob += prob
            else:
                children_without_prob += 1

        if total_prob > 1.0:
            raise ValueError(f"Total probability exceeds 1.0: {total_prob}")

        for child, prob in children.items():
            if prob is None:
                children[child] = (1.0 - total_prob) / children_without_prob

        return children


def generate_dataset_matrix(
        attributes: list[Enum],
        relations: list[Relation],
        size: int = 1_000
        ) -> list[list[Enum]]:
    # This function generates a matrix (e.g. a CSV file) containing all possible attribute combinations
    # The probability distribution of attribute occurrences in records follows the relationships between attributes
    # Example for
    # * Attributes: Cat, Dog and Colors: Red, Green
    # * Relations: Cat: Red 0.7, Green 0.3
    # * n = 100
    # Result:
    # * 35 records: Cat, Red
    # * 15 records: Cat, Green
    # * 25 records: Dog, Red
    # * 25 records: Dog, Green
    if size < 0:
        return [[]]

    if not attributes:
        return []

    attribute_types = list(attributes)
    attribute_members: dict[type[Enum], list[Enum]] = {}
    base_probabilities: dict[type[Enum], dict[Enum, float]] = {}

    for attribute_type in attribute_types:
        members = list(attribute_type)
        if not members:
            raise ValueError(f"Attribute '{attribute_type.__name__}' must contain at least one value")
        attribute_members[attribute_type] = members
        uniform_probability = 1.0 / len(members)
        base_probabilities[attribute_type] = {
            member: uniform_probability for member in members
        }

    relation_map: dict[Enum, dict[type[Enum], dict[Enum, float]]] = {}

    for relation in relations:
        parent_type = relation.parent.__class__
        if parent_type not in attribute_members:
            raise ValueError(
                f"Relation parent '{relation.parent}' does not belong to provided attributes"
            )

        if not relation.children:
            raise ValueError(f"Relation for '{relation.parent}' must define at least one child")

        child_type = next(iter(relation.children)).__class__
        if child_type not in attribute_members:
            raise ValueError(
                f"Relation children for '{relation.parent}' do not belong to provided attributes"
            )

        known_children = set(attribute_members[child_type])
        if set(relation.children) != known_children:
            raise ValueError(
                f"Relation for '{relation.parent}' must define probabilities for all values of "
                f"'{child_type.__name__}'"
            )

        parent_relations = relation_map.setdefault(relation.parent, {})
        if child_type in parent_relations:
            raise ValueError(
                f"Duplicate relation defined for parent '{relation.parent}' and child type "
                f"'{child_type.__name__}'"
            )

        parent_relations[child_type] = relation.children

    weighted_rows: list[tuple[list[Enum], float]] = []

    for combination in product(*(attribute_members[attribute_type] for attribute_type in attribute_types)):
        row = list(combination)
        row_by_type = {
            attribute_type: value
            for attribute_type, value in zip(attribute_types, combination, strict=True)
        }

        probability = 1.0
        for attribute_type, value in row_by_type.items():
            probability *= base_probabilities[attribute_type][value]

        for parent_value, child_relations in relation_map.items():
            parent_type = parent_value.__class__
            if row_by_type[parent_type] != parent_value:
                continue

            for child_type, child_probabilities in child_relations.items():
                child_value = row_by_type[child_type]
                base_probability = base_probabilities[child_type][child_value]
                if base_probability <= 0:
                    raise ValueError(
                        f"Base probability for '{child_type.__name__}.{child_value.name}' must be positive"
                    )
                probability *= child_probabilities[child_value] / base_probability

        weighted_rows.append((row, probability))

    total_probability = sum(probability for _, probability in weighted_rows)
    if total_probability <= 0:
        raise ValueError("Total probability must be greater than 0")

    normalized_rows = [
        (row, probability / total_probability)
        for row, probability in weighted_rows
    ]

    raw_counts = [probability * size for _, probability in normalized_rows]
    counts = [int(raw_count) for raw_count in raw_counts]
    remaining = size - sum(counts)

    remainders = sorted(
        (
            (raw_count - count, index)
            for index, (raw_count, count) in enumerate(zip(raw_counts, counts, strict=True))
        ),
        reverse=True,
    )

    for _, index in remainders[:remaining]:
        counts[index] += 1

    dataset_matrix: list[list[Enum]] = []
    for (row, _), count in zip(normalized_rows, counts, strict=True):
        dataset_matrix.extend([list(row) for _ in range(count)])

    return dataset_matrix


def weighted_choice(rng: Random, probabilities: dict[Enum, float]) -> Enum:
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("Probabilities must sum to > 0")

    r = rng.random() * total
    acc = 0.0

    for value, prob in probabilities.items():
        acc += prob
        if r <= acc:
            return value

    return list(probabilities.keys())[-1]


def generate_dataset_cyclic(
    attributes: list[Type[Enum]],
    relations: list[Relation],
    size: int = 1000,
    burn_in: int = 100,
    thinning: int = 1,
    seed: int | None = None,
) -> list[list[Enum]]:
    """
    Generator for cyclic relations.

    Relations are treated as factors of the joint distribution:
        P(A, B, C, D) ∝ P(B|A) P(C|B) P(D|C) P(A|D)

    For each attribute combination, a weight is computed as the product
    of conditional probabilities defined by the relations. The weights are
    then normalized and converted into exactly ``size`` records.

    The ``burn_in`` and ``thinning`` parameters remain in the signature
    for backward compatibility, but they are not used. This generator is
    not a Markov process and does not depend on the order of relation updates.
    """

    if size <= 0:
        return []

    rng = Random(seed)

    attribute_members: dict[Type[Enum], list[Enum]] = {}
    for attr_type in attributes:
        members = list(attr_type)
        if not members:
            raise ValueError(f"Attribute {attr_type.__name__} has no values")
        attribute_members[attr_type] = members

    for relation in relations:
        parent_type = relation.parent.__class__
        if parent_type not in attribute_members:
            raise ValueError(f"Unknown parent type: {parent_type.__name__}")

        child_values = list(relation.children.keys())
        if not child_values:
            raise ValueError(f"Relation for {relation.parent} has no children")

        child_type = child_values[0].__class__

        if child_type not in attribute_members:
            raise ValueError(f"Unknown child type: {child_type.__name__}")

        if any(child.__class__ is not child_type for child in child_values):
            raise ValueError("All children in one relation must have the same Enum type")

        expected = set(attribute_members[child_type])
        actual = set(child_values)

        if actual != expected:
            raise ValueError(
                f"Relation {relation.parent} must define probabilities "
                f"for all values of {child_type.__name__}"
            )

        prob_sum = sum(relation.children.values())
        if abs(prob_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Probabilities for {relation.parent} must sum to 1. "
                f"Got {prob_sum}"
            )

    weighted_rows: list[tuple[list[Enum], float]] = []

    for combination in product(*(attribute_members[attr_type] for attr_type in attributes)):
        row = list(combination)
        row_by_type = {
            attr_type: value
            for attr_type, value in zip(attributes, combination, strict=True)
        }

        weight = 1.0
        for relation in relations:
            parent_type = relation.parent.__class__
            if row_by_type[parent_type] != relation.parent:
                continue

            child_type = next(iter(relation.children)).__class__
            child_value = row_by_type[child_type]
            weight *= relation.children[child_value]

        weighted_rows.append((row, weight))

    total_weight = sum(weight for _, weight in weighted_rows)
    if total_weight <= 0:
        raise ValueError("Total probability must be greater than 0")

    normalized_rows = [
        (row, weight / total_weight)
        for row, weight in weighted_rows
    ]

    raw_counts = [probability * size for _, probability in normalized_rows]
    counts = [int(raw_count) for raw_count in raw_counts]
    remaining = size - sum(counts)

    remainders = sorted(
        (
            (raw_count - count, index)
            for index, (raw_count, count) in enumerate(zip(raw_counts, counts, strict=True))
        ),
        reverse=True,
    )

    for _, index in remainders[:remaining]:
        counts[index] += 1

    dataset: list[list[Enum]] = []
    for (row, _), count in zip(normalized_rows, counts, strict=True):
        dataset.extend([list(row) for _ in range(count)])

    rng.shuffle(dataset)
    return dataset


def generate_wild_rows(
        relations: list[Relation],
        size: int = 1_000,
        attributes: list[type[Enum]] | None = None,
        subset: str = "wild",
        interaction: str = "chasing",
        environment: str = "field",
        pose: str = "running",
        ) -> list[dict[str, str]]:
    if attributes is None:
        attributes = [
            WildPredator,
            WildPredatorColor,
            WildPrey,
            WildPreyColor,
            TimeOfDay,
            wild_animals.WildSeason,
        ]

    dataset = generate_dataset_matrix(attributes, relations, size=size)
    rows: list[dict[str, str]] = []

    for sample in dataset:
        row = {column: "none" for column in DATASET_COLUMNS}
        row["image_path"] = ""
        row["subset"] = subset
        row["interaction"] = interaction
        row["environment"] = environment
        row["pose"] = pose

        for value in sample:
            column = ATTRIBUTE_COLUMN_MAP.get(value.__class__)
            if column is None:
                raise ValueError(
                    f"Unsupported attribute type for CSV export: {value.__class__.__name__}"
                )
            row[column] = _csv_value(value)

        rows.append(row)

    return rows


def generate_wild_csv(
        output_path: str,
        relations: list[Relation],
        size: int = 1_000,
        attributes: list[type[Enum]] | None = None,
        subset: str = "wild",
        interaction: str = "chasing",
        environment: str = "field",
        pose: str = "running",
        ) -> list[dict[str, str]]:
    rows = generate_wild_rows(
        relations=relations,
        size=size,
        attributes=attributes,
        subset=subset,
        interaction=interaction,
        environment=environment,
        pose=pose,
    )

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def generate_domestic_rows(
        relations: list[Relation],
        size: int = 1_000,
        attributes: list[type[Enum]] | None = None,
        subset: str = "domestic",
        ) -> list[dict[str, str]]:
    if attributes is None:
        attributes = [
            PetAnimal,
            PetColor,
            PetPosition,
            PetRoom,
            TimeOfDay,
            BallObject,
            BallColor,
        ]

    dataset = generate_dataset_matrix(attributes, relations, size=size)
    rows: list[dict[str, str]] = []

    for animal, animal_color, position, room, time_of_day, ball_object, ball_color in dataset:
        has_ball = ball_object == BallObject.BALL
        row = {column: "none" for column in DATASET_COLUMNS}
        row["image_path"] = ""
        row["subset"] = subset
        row["object_1"] = _csv_value(animal)
        row["object_1_color"] = _csv_value(animal_color)
        row["object_2"] = _csv_value(ball_object) if has_ball else "none"
        row["object_2_color"] = _csv_value(ball_color) if has_ball else "none"
        row["interaction"] = "playing_with_ball" if has_ball else "not_playing_with_ball"
        row["environment"] = "living_room" if room == PetRoom.LIVING_ROOM else "kitchen"
        row["pose"] = _csv_value(position)
        row["day_time"] = _csv_value(time_of_day)
        row["season"] = "none"
        rows.append(row)

    return rows


def generate_birds_rows(
        relations: list[Relation],
        size: int = 1_000,
        attributes: list[type[Enum]] | None = None,
        subset: str = "birds",
        ) -> list[dict[str, str]]:
    if attributes is None:
        attributes = [
            BirdColor,
            BirdPosition,
            TimeOfDay,
            BeakState,
        ]

    dataset = generate_dataset_matrix(attributes, relations, size=size)
    rows: list[dict[str, str]] = []

    for bird_color, position, time_of_day, beak_state in dataset:
        has_worm = beak_state == BeakState.WORM
        row = {column: "none" for column in DATASET_COLUMNS}
        row["image_path"] = ""
        row["subset"] = subset
        row["object_1"] = "bird"
        row["object_1_color"] = _csv_value(bird_color)
        row["object_2"] = "worm" if has_worm else "none"
        row["object_2_color"] = "none"
        row["interaction"] = "holding_worm_in_beak" if has_worm else "none"
        row["environment"] = BIRD_ENVIRONMENT_MAP[position]
        row["pose"] = BIRD_POSE_MAP[position]
        row["day_time"] = _csv_value(time_of_day)
        row["season"] = "none"
        rows.append(row)

    return rows


def generate_all_subsets_csv(
        output_path: str,
        *,
        wild_relations: list[Relation],
        domestic_relations: list[Relation] | None = None,
        birds_relations: list[Relation] | None = None,
        wild_size: int = 1_000,
        domestic_size: int = 1_000,
        birds_size: int = 1_000,
        ) -> list[dict[str, str]]:
    all_rows = []
    all_rows.extend(generate_wild_rows(wild_relations, size=wild_size))

    if domestic_relations is None:
        domestic_relations = [
            Relation(PetAnimal.DOG, {
                PetRoom.LIVING_ROOM: 0.6,
                PetRoom.KITCHEN: 0.4,
            }),
            Relation(PetAnimal.CAT, {
                PetRoom.LIVING_ROOM: 0.3,
                PetRoom.KITCHEN: 0.7,
            }),
            Relation(PetPosition.LYING, {
                BallObject.BALL: 0.55,
                BallObject.NONE: 0.45,
            }),
            Relation(PetPosition.STANDING, {
                BallObject.BALL: 0.75,
                BallObject.NONE: 0.25,
            }),
            Relation(PetPosition.LYING, {
                BallColor.RED: 0.35,
                BallColor.YELLOW: 0.2,
                BallColor.GREEN: 0.15,
                BallColor.BLUE: 0.1,
                BallColor.ORANGE: 0.1,
                BallColor.PURPLE: 0.1,
            }),
            Relation(PetPosition.STANDING, {
                BallColor.RED: 0.1,
                BallColor.YELLOW: 0.15,
                BallColor.GREEN: 0.2,
                BallColor.BLUE: 0.2,
                BallColor.ORANGE: 0.15,
                BallColor.PURPLE: 0.2,
            }),
        ]
    all_rows.extend(generate_domestic_rows(domestic_relations, size=domestic_size))

    if birds_relations is None:
        birds_relations = [
            Relation(BirdPosition.FLYING, {
                BeakState.EMPTY: 0.85,
                BeakState.WORM: 0.15,
            }),
            Relation(BirdPosition.NEST, {
                BeakState.EMPTY: 0.4,
                BeakState.WORM: 0.6,
            }),
            Relation(BirdPosition.BRANCH, {
                BeakState.EMPTY: 0.65,
                BeakState.WORM: 0.35,
            }),
        ]
    all_rows.extend(generate_birds_rows(birds_relations, size=birds_size))

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    return all_rows


if __name__ == "__main__":
    attributes = [WildPredator, WildPrey, WildPredatorColor, WildPreyColor]

    relations = [
        Relation(WildPredator.FOX, {
            WildPredatorColor.RED: 0.4,
            WildPredatorColor.WHITE: 0.3,
            WildPredatorColor.BLACK: 0.2,
            WildPredatorColor.BROWN: 0.1
        }),
        Relation(WildPrey.MOUSE, {
            WildPredatorColor.RED: 0.1,
            WildPredatorColor.WHITE: 0.2,
            WildPredatorColor.BLACK: 0.3,
            WildPredatorColor.BROWN: 0.4
        }),
    ]


    attributes = [WildPredator, WildPrey]
    attributes = [
        wild_animals.WildPredator, wild_animals.WildPredatorColor,
        wild_animals.WildPrey, wild_animals.WildPreyColor,
        TimeOfDay,
        wild_animals.WildSeason
        ]

    relations = [
        Relation(WildPredator.FOX, {
            WildPrey.HARE: 0.25,
            WildPrey.MOUSE: 0.75
        }),
        Relation(WildPredator.DOG, {
            WildPrey.HARE: 0.76,
            WildPrey.MOUSE: 0.24
        }),
        Relation(WildPredator.CAT, {
            WildPrey.HARE: 0.5,
            WildPrey.MOUSE: 0.5
        }),
    ]

    rows = generate_all_subsets_csv(
        "dataset.csv",
        wild_relations=relations,
        wild_size=10_000,
        domestic_size=10_000,
        birds_size=10_000,
    )
