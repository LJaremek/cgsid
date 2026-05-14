from __future__ import annotations

from cgsid.core.labels import LabelSpec, enum_label_specs, label_specs
from cgsid.enums import PetAnimal, PetPosition, PetRoom, TimeOfDay, WildPredator, WildPrey, WildSeason


OBJECT_PRESENCE_LABELS = (
    ("animal", "a photo containing an animal"),
    ("bird", "a photo containing a bird"),
    ("dog", "a photo containing a dog"),
    ("cat", "a photo containing a cat"),
    ("fox", "a photo containing a fox"),
    ("hare", "a photo containing a hare"),
    ("mouse", "a photo containing a mouse"),
    ("ball", "a photo containing a ball"),
    ("worm", "a photo containing a worm"),
)

SCENE_TYPE_LABELS = (
    ("one_animal", "a photo containing one animal"),
    ("two_animals", "a photo containing two animals"),
    ("indoor_animal_scene", "a photo containing an indoor animal scene"),
    ("wildlife_scene", "a photo containing a wildlife scene"),
)

GENERAL_POSE_LABELS = (
    ("flying", "a photo of an animal flying"),
    ("sitting_in_nest", "a photo of an animal sitting in a nest"),
    ("standing_on_branch", "a photo of an animal standing on a branch"),
    ("lying_down", "a photo of an animal lying down"),
    ("standing", "a photo of an animal standing"),
    ("running", "a photo of an animal running"),
)

GENERAL_INTERACTION_LABELS = (
    ("playing_with_ball", "a photo of an animal playing with a ball"),
    ("not_playing_with_ball", "a photo of an animal not playing with a ball"),
    ("chasing_interaction", "a photo of one animal chasing another animal"),
    ("holding_worm_in_beak", "a photo of a bird holding a visible worm between the tips of its beak"),
    ("empty_beak", "a photo of a bird with an empty beak and no worm or food"),
)

GENERAL_ENVIRONMENT_LABELS = (
    ("living_room", "a photo taken in a living room"),
    ("kitchen", "a photo taken in a kitchen"),
    ("open_field", "a photo taken in an open field"),
    ("sky", "a photo taken in the sky"),
    ("nest", "a photo taken near a nest"),
    ("tree_branch", "a photo taken on a tree branch"),
)

COLOR_PRESENCE_LABELS = (
    ("dark", ("a photo of an animal with dark fur", "the animal is dark colored", "a dark colored pet or animal")),
    ("brown", ("a photo of an animal with brown fur", "the animal is brown", "a brown pet or animal")),
    ("white", ("a photo of an animal with white fur", "the animal is white", "a white pet or animal")),
    ("ginger", ("a photo of an animal with ginger or orange fur", "the animal is ginger colored", "a ginger cat or pet")),
    ("gray", ("a photo of an animal with gray fur", "the animal is gray", "a gray pet or animal")),
    ("black", ("a photo of an animal with black fur", "the animal is black", "a black pet or animal")),
    ("blue", ("a photo of an animal with blue feathers or coloring", "the animal is blue", "a blue animal")),
    ("red", ("a photo of an animal with red feathers or coloring", "the animal is red", "a red animal")),
    ("yellow", ("a photo of an animal with yellow feathers or coloring", "the animal is yellow", "a yellow animal")),
    ("green", ("a photo of an animal with green feathers or coloring", "the animal is green", "a green animal")),
)

BALL_COLOR_LABELS = (
    ("red", ("a photo containing a red ball", "the ball is red", "a pet toy ball colored red")),
    ("yellow", ("a photo containing a yellow ball", "the ball is yellow", "a pet toy ball colored yellow")),
    ("green", ("a photo containing a green ball", "the ball is green", "a pet toy ball colored green")),
    ("blue", ("a photo containing a blue ball", "the ball is blue", "a pet toy ball colored blue")),
    ("orange", ("a photo containing an orange ball", "the ball is orange", "a pet toy ball colored orange")),
    ("purple", ("a photo containing a purple ball", "the ball is purple", "a pet toy ball colored purple")),
)


def build_domestic_label_specs() -> list[LabelSpec]:
    labels: list[LabelSpec] = []
    labels.extend(build_shared_label_specs())
    labels.extend(build_domestic_specific_label_specs())
    labels.extend(build_wild_reference_label_specs())
    return labels


def build_shared_label_specs() -> list[LabelSpec]:
    labels: list[LabelSpec] = []
    labels.extend(label_specs(values=OBJECT_PRESENCE_LABELS, subset="all", group="object_presence", key_prefix="contains"))
    labels.extend(label_specs(values=SCENE_TYPE_LABELS, subset="all", group="scene_type", key_prefix="scene"))
    labels.extend(label_specs(values=GENERAL_POSE_LABELS, subset="all", group="pose", key_prefix="pose"))
    labels.extend(label_specs(values=GENERAL_INTERACTION_LABELS, subset="all", group="interaction", key_prefix="interaction"))
    labels.extend(label_specs(values=GENERAL_ENVIRONMENT_LABELS, subset="all", group="environment", key_prefix="environment"))
    labels.extend(
        enum_label_specs(
            values=TimeOfDay,
            subset="all",
            group="time_of_day",
            key_prefix="time",
            text_builder=lambda value: f"a photo taken during {value.value}",
        )
    )
    labels.extend(
        enum_label_specs(
            values=WildSeason,
            subset="all",
            group="season",
            key_prefix="season",
            text_builder=lambda season: f"a photo taken in {season.value}",
        )
    )
    labels.extend(label_specs(values=COLOR_PRESENCE_LABELS, subset="all", group="color_presence", key_prefix="contains_color"))
    labels.extend(label_specs(values=BALL_COLOR_LABELS, subset="all", group="ball_color", key_prefix="contains_ball_color"))
    return labels


def build_domestic_specific_label_specs() -> list[LabelSpec]:
    labels: list[LabelSpec] = []
    labels.extend(
        label_specs(
            values=(
                (
                    PetAnimal.DOG.value,
                    (
                        "a photo of a dog indoors",
                        "a domestic dog in a room",
                        "the pet is a dog, not a cat",
                    ),
                ),
                (
                    PetAnimal.CAT.value,
                    (
                        "a photo of a cat indoors",
                        "a domestic cat in a room",
                        "the pet is a cat, not a dog",
                    ),
                ),
            ),
            subset="domestic",
            group="animal",
            key_prefix="pet_animal",
        )
    )
    labels.extend(
        label_specs(
            values=(
                (
                    PetPosition.LYING.value,
                    (
                        "a photo of a pet lying down on the floor",
                        "the pet is lying on the floor",
                        "a dog or cat resting with its body low to the floor",
                    ),
                ),
                (
                    PetPosition.STANDING.value,
                    (
                        "a photo of a pet standing on the floor",
                        "the pet is standing upright on its legs",
                        "a dog or cat standing, not lying down",
                    ),
                ),
            ),
            subset="domestic",
            group="pose",
            key_prefix="pet_position",
        )
    )
    labels.extend(
        label_specs(
            values=(
                (
                    PetRoom.LIVING_ROOM.value,
                    (
                        "a photo of a pet in a living room",
                        "an indoor pet scene in a living room",
                        "a dog or cat in a lounge with living room furniture",
                    ),
                ),
                (
                    PetRoom.KITCHEN.value,
                    (
                        "a photo of a pet in a kitchen",
                        "an indoor pet scene in a kitchen",
                        "a dog or cat in a kitchen with cabinets or appliances",
                    ),
                ),
            ),
            subset="domestic",
            group="environment",
            key_prefix="pet_room",
        )
    )
    labels.extend(
        label_specs(
            values=(
                (
                    TimeOfDay.DAY.value,
                    (
                        "a bright indoor pet photo lit by natural daylight from windows",
                        "a dog or cat indoors during daytime with neutral white daylight",
                        "a pet scene with clear daylight and no warm evening lamp lighting",
                    ),
                ),
                (
                    TimeOfDay.EVENING.value,
                    (
                        "an indoor pet photo in the evening lit by warm household lamps",
                        "a dog or cat indoors at evening with warm artificial light and darker windows",
                        "a cozy evening pet scene with warm lamp light and no bright daytime sunlight",
                    ),
                ),
            ),
            subset="domestic",
            group="time_of_day",
            key_prefix="pet_time",
        )
    )
    labels.extend(
        label_specs(
            values=(
                (
                    "playing_with_ball",
                    (
                        "a photo of a pet playing with a ball",
                        "a dog or cat interacting with a ball",
                        "a pet next to and playing with a visible ball",
                    ),
                ),
                (
                    "not_playing_with_ball",
                    (
                        "a photo of a pet without a ball",
                        "a dog or cat not playing with a ball",
                        "a pet scene where no ball is being played with",
                    ),
                ),
            ),
            subset="domestic",
            group="interaction_role",
            key_prefix="pet_state",
        )
    )
    return labels


def build_wild_reference_label_specs() -> list[LabelSpec]:
    labels: list[LabelSpec] = []
    labels.extend(
        enum_label_specs(
            values=WildPredator,
            subset="wild",
            group="predator",
            key_prefix="wild_predator",
            text_builder=lambda predator: f"a photo of a {predator.value} chasing another animal",
        )
    )
    labels.extend(
        enum_label_specs(
            values=WildPrey,
            subset="wild",
            group="prey",
            key_prefix="wild_prey",
            text_builder=lambda prey: f"a photo of a wild animal chasing a {prey.value}",
        )
    )
    labels.extend(
        label_specs(
            values=(
                ("animal_chasing", "a photo of an animal chasing"),
                ("animal_being_chased", "a photo of an animal being chased"),
            ),
            subset="wild",
            group="interaction_role",
            key_prefix="wild_state",
        )
    )
    for predator in WildPredator:
        for prey in WildPrey:
            labels.append(
                LabelSpec(
                    key=f"wild_pair_{predator.value.replace(' ', '_')}_{prey.value.replace(' ', '_')}",
                    texts=f"a photo of a {predator.value} chasing a {prey.value}",
                    subset="wild",
                    group="pair",
                )
            )
    return labels
