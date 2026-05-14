from __future__ import annotations

import random

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


def build_wild_animal_prompt(
    chasing_animal: WildPredator,
    chased_animal: WildPrey,
    time_of_day: TimeOfDay,
    season: WildSeason,
    chasing_animal_color: WildPredatorColor,
    chased_animal_color: WildPreyColor,
    camera_angle: CameraAngle,
    camera_distance: CameraDistance,
) -> str:
    distance_phrase = random.choice(
        [
            f"The distance between the {chasing_animal.value} and the {chased_animal.value} is very short, with the predator nearly reaching the prey in the same stride.",
            f"There is roughly one body length of clear space between the {chasing_animal.value} and the {chased_animal.value}, making the pursuit feel immediate and dangerous.",
            f"The {chasing_animal.value} is close behind the {chased_animal.value}, separated by only a small but visible gap that emphasizes the tension of the chase.",
        ]
    )
    angle_phrase = random.choice(
        [
            f"The photograph is taken from a {camera_angle.value} at a {camera_distance.value}, giving a dynamic view of the chase.",
            f"The camera observes the action from a {camera_angle.value} using a {camera_distance.value} framing, keeping the pursuit tense and readable.",
            f"The scene is captured from a {camera_angle.value} with a {camera_distance.value} composition, clearly emphasizing the direction and intensity of the chase.",
        ]
    )

    if season == WildSeason.WINTER:
        scene_block = f"""
        The image shows a {chasing_animal_color.value} {chasing_animal.value} in the middle of a hunting action, pursuing a {chased_animal_color.value} {chased_animal.value} across a snowy winter field.
        """.strip()
        season_environment_block = """
        * The surrounding environment is a fully wintry open field or meadow, with the ground predominantly covered in visible snow.
        * Snow should be the dominant ground surface across the whole scene, not just small patches.
        * The chase should disturb the snow, with believable footprints, kicked-up powder, compressed tracks, or scattered snow behind the animals.
        * Any visible vegetation must be sparse, dry, frost-covered, partially buried, or dormant. There must be no lush green grass and no summer-like meadow appearance.
        * The color palette should feel cold and wintery, with whites, muted browns, pale grays, and crisp seasonal contrast.
        * The wider atmosphere should clearly communicate winter through cold light, frosty air, and believable snow-season landscape details.
        """.strip()
    else:
        scene_block = f"""
        The image shows a {chasing_animal_color.value} {chasing_animal.value} in the middle of a hunting action, pursuing a {chased_animal_color.value} {chased_animal.value} across an open summer meadow.
        """.strip()
        season_environment_block = """
        * The surrounding environment is an open summer meadow with short grass and naturally uneven ground, without major obstacles between the two animals.
        * The field should feel dry to mildly lush depending on the light, but overall clearly warm-season and snow-free.
        * The wider atmosphere should communicate summer naturally through the vegetation, warmth of the light, and believable seasonal landscape details.
        """.strip()

    if time_of_day == TimeOfDay.DAY:
        time_atmosphere_block = """
        * The time of day must read as clear daytime: bright natural sunlight, a visibly bright sky, crisp daylight contrast, and readable daylight shadows on the ground.
        * The scene must not look like sunset, dusk, night, or warm lamp-like evening light.
        """.strip()
    else:
        time_atmosphere_block = """
        * The time of day must read as late evening after sunset, not daytime and not bright golden hour.
        * The sky should be deep navy blue, purple, or dark blue-gray, with only a faint sunset afterglow near the horizon.
        * The scene should be visibly dim and low-light, with darker ground exposure, soft shadowed forms, silhouettes or rim light, and muted colors.
        * The animals must remain recognizable, but the overall image should feel moody, dusky, and clearly under evening darkness.
        * There must be no bright blue sky, no overhead sunlight, no sunny meadow look, no crisp midday illumination, and no scene that could be mistaken for daytime.
        * The late-evening darkness should be unambiguous even at small image size.
        """.strip()

    return f"""
        {scene_block}

        Lighting and time of day:
        {time_atmosphere_block}

        {angle_phrase}

        The {chasing_animal.value} is positioned slightly to the left side of the frame, its body stretched forward in a low, dynamic leap. Its hind legs are extended backward, pushing off the ground, while its front legs are lifted and bent slightly, preparing to land. The {chasing_animal.value}'s body is elongated horizontally, parallel to the ground, indicating forward motion. Its head is lowered and aligned with its spine, pointing directly toward the {chased_animal.value}. The ears are upright and angled forward, signaling intense focus. Its eyes are fixed on the {chased_animal.value}, and its muzzle is slightly open, revealing tension and concentration. The tail is extended straight behind it, slightly elevated, acting as a counterbalance.

        The {chased_animal.value} is positioned to the right of the {chasing_animal.value}, a short distance ahead. {distance_phrase} Its body is angled diagonally toward the right edge of the frame, suggesting it is attempting to escape in that direction. The {chased_animal.value}'s hind legs are fully extended backward in mid-stride, while its front legs are reaching forward toward the ground, indicating a rapid bounding motion. Its body is slightly arched, and its head is turned marginally forward and to the right, focused on its escape path rather than looking back at the predator. The ears are swept slightly backward due to speed and tension.

        Relative positioning:
        * The {chasing_animal.value} is directly behind the {chased_animal.value}, aligned almost in a straight pursuit line.
        * The {chasing_animal.value}'s head is aimed precisely at the {chased_animal.value}'s hindquarters.
        * {distance_phrase}
        * The {chased_animal.value} is lower to the ground and closer to the right edge of the image.
        * The {chasing_animal.value} occupies the central-left portion, moving rightward.
        * Both animals are oriented from left to right across the frame.
        {season_environment_block}
        * The scene takes place during {time_of_day.value} in {season.value}. The lighting and atmosphere reflect this naturally.
        * The generated image must match the requested season unambiguously. Do not depict green summer grass when the season is winter.

        Overall, the composition captures a moment of high tension: the {chasing_animal.value} advancing rapidly from behind, fully focused and streamlined, while the {chased_animal.value} flees forward in a desperate, powerful leap toward the right side of the scene.
    """.strip()


def build_flux_pet_prompt(
    animal: PetAnimal,
    animal_color: PetColor,
    position: PetPosition,
    is_playing_with_ball: bool,
    room: PetRoom,
    time_of_day: TimeOfDay,
    ball_object: BallObject | None = None,
    ball_color: BallColor | None = None,
) -> str:
    if is_playing_with_ball and ball_object is None:
        raise ValueError("ball_object is required when is_playing_with_ball=True")
    if is_playing_with_ball and ball_color is None:
        raise ValueError("ball_color is required when is_playing_with_ball=True")

    if position == PetPosition.LYING:
        position_block = """
            The animal is lying naturally on the floor, with its body fully supported by the ground in a relaxed but believable posture. Its front paws are placed naturally in front of the body or slightly to the sides, and its torso is clearly visible. The pose should look anatomically correct and comfortable, like a real pet resting indoors.
            """.strip()
    else:
        position_block = """
            The animal is standing naturally on all four legs, with a balanced, realistic posture. Its body weight is distributed in a believable way, the legs are placed correctly, and the spine and head position look natural for a real pet indoors. The full standing pose should be clearly visible.
            """.strip()

    if is_playing_with_ball:
        interaction_block = f"""
            The {animal.value} is actively playing with a {ball_color.value} ball. The ball is clearly visible close to the animal and the interaction is unambiguous. The pet's attention is focused on the ball, with the head and eyes directed toward it. The body language should communicate playful curiosity and engagement. The paws or muzzle may be near the ball, suggesting active play, but the anatomy and pose must remain realistic and natural.
            """.strip()
        ball_visibility = f"""
            The {ball_color.value} ball should be easy to identify in the frame, placed naturally on the floor and integrated into the scene as the main object of interaction.
            """.strip()
    else:
        interaction_block = f"""
            There is no ball in the scene and the {animal.value} is not playing with any toy. The focus is entirely on the pet itself, its posture, expression, and presence within the indoor environment.
            """.strip()
        ball_visibility = ""

    if room == PetRoom.LIVING_ROOM:
        room_block = """
            The scene takes place in a realistic living room. The environment should include natural interior details such as a sofa, carpet, wooden floor or panels, coffee table, soft household decor, and a lived-in but tidy atmosphere. The room should look believable, comfortable, and visually coherent, like a real home interior.
            """.strip()
    else:
        room_block = """
            The scene takes place in a realistic kitchen. The environment should include natural kitchen details such as cabinets, countertops, tiled or wooden floor, household appliances, and subtle everyday objects. The kitchen should feel authentic, lived-in, and visually coherent, like a real home interior.
            """.strip()

    if time_of_day == TimeOfDay.DAY:
        light_block = """
            The lighting clearly and unambiguously indicates daytime. Bright natural daylight enters through visible windows or an obvious daylight source, with pale blue or white daylight outside the window, neutral white illumination, soft realistic daylight shadows, and clear visibility of the pet, the floor, and the surrounding furniture. The room should not look like night, dusk, or warm lamp-lit evening.
            """.strip()
    else:
        light_block = """
            The lighting clearly and unambiguously indicates late evening after sunset. The room is dim and cozy, mainly lit by warm indoor lamps, household light, or small practical lights, while any visible windows are dark or show deep blue/purple twilight outside. The pet remains visible, but the room should have obvious evening darkness, soft shadows, warm lamp pools, and a low-light indoor atmosphere. The scene must not look like bright midday daylight, a sunlit room, or a room with pale daytime window light.
            """.strip()

    return f"""
        Photorealistic indoor pet photography of a {animal_color.value} {animal.value} in a home {room.value}.

        Lighting and atmosphere:
        {light_block}

        Main subject:
        A realistic {animal_color.value} {animal.value} is the clear main subject of the image. The animal should have natural anatomy, realistic proportions, detailed fur texture, believable ears, paws, eyes, muzzle, and body shape. The animal must have exactly four legs, no more and no fewer. The fur color must clearly read as {animal_color.value}, with realistic tonal variation and natural-looking texture.

        Body pose:
        {position_block}

        Interaction:
        {interaction_block}
        {ball_visibility}

        Environment:
        {room_block}

        Composition:
        The animal should be fully visible in the frame and remain the dominant subject. The camera should be positioned at pet eye level or slightly below eye level to create a natural and intimate perspective. The framing should feel like a real candid indoor photograph. The background should provide enough environmental context to clearly communicate the {room.value}, but it should not distract from the animal. Depth of field should be natural, with the pet rendered sharply and the background slightly softer if appropriate.

        Style and quality:
        Highly detailed, photorealistic, realistic pet photography, natural indoor scene, natural posture, accurate anatomy, realistic fur, believable home interior, visually coherent composition, soft realistic shadows, high detail, cinematic but natural, documentary-like realism, Ultra HD, 4K.
        """.strip()


def build_flux_bird_prompt(
    bird_color: BirdColor,
    position: BirdPosition,
    time_of_day: TimeOfDay,
    beak_state: BeakState,
) -> str:
    if position == BirdPosition.FLYING:
        position_block = """
            The bird is captured mid-flight with wings extended. The body is streamlined and the wings show a natural aerodynamic shape. The feathers are slightly spread and the posture clearly indicates active flight through the air.
            """.strip()
    elif position == BirdPosition.NEST:
        position_block = """
            The bird is sitting calmly inside a natural nest made of small branches and twigs. Its body is slightly lowered into the nest and its wings are folded along its sides. The posture suggests the bird is resting or guarding the nest.
            """.strip()
    else:
        position_block = """
            The bird is standing on a natural tree branch. Its claws grip the branch securely and its body posture is upright and balanced. The wings are folded naturally along the body.
            """.strip()

    if time_of_day == TimeOfDay.DAY:
        light_block = """
            The lighting indicates bright daytime conditions. The background must include clearly visible bright blue sky or pale daylight sky, with natural sunlight illuminating the bird and surroundings. Realistic daylight shadows and crisp visibility should make the time of day obvious even in a small image.
            """.strip()
    else:
        light_block = """
            The lighting indicates late evening after sunset, in dim blue-hour or dusk conditions. The background must include a deep navy blue, purple, or dark blue-gray twilight sky, with at most a faint orange/pink afterglow near the horizon. The bird should remain visible through soft low-light exposure, subtle rim light, or shadowed detail, but the overall scene must clearly feel dark, dusky, and moody. There must be no bright blue midday sky, no overhead sunlight, no sunny daytime exposure, and no scene that could be mistaken for daytime.
            """.strip()

    if beak_state == BeakState.WORM:
        beak_block = """
            The bird is holding a small worm in its beak. The worm must be clearly visible even in a small 224x224 image, with part of the worm protruding from between the beak tips. The beak should visibly grip the worm, and the worm should contrast against the bird's feathers and the background. The detail should read as food held in the beak, not as a twig, shadow, feather, or random mark.
            """.strip()
    else:
        beak_block = """
            The bird's beak is empty and clearly visible. There must be no worm, insect, twig, food, or object between the beak tips. The beak shape and texture should appear natural and anatomically correct.
            """.strip()

    return f"""
        Photorealistic wildlife photograph of a {bird_color.value} bird.

        Lighting and atmosphere:
        {light_block}

        Main subject:
        A realistic {bird_color.value} bird is the main subject of the image. The bird should have natural anatomy, realistic feather structure, believable wings, eyes, and beak. The feather color clearly appears {bird_color.value} with natural tonal variation and fine feather texture.

        Body pose:
        {position_block}

        Beak detail:
        {beak_block}

        Environment:
        The surrounding environment should feel natural and believable for a bird habitat. The sky must be visible in the background for every pose so the time of day can be recognized clearly. If the bird is flying, the background should show open sky with natural atmospheric depth. If the bird is in a nest or on a branch, the scene should include realistic tree elements such as bark texture, leaves, and branches, but with visible sky behind or between the branches. For evening images, that visible sky must be dark blue-hour or dusk sky, not daytime.

        Composition:
        The bird should be clearly visible and remain the dominant subject of the image. The camera framing should resemble wildlife photography, with a natural perspective and slightly softened background to emphasize the bird.

        Style and quality:
        Highly detailed, photorealistic wildlife photography, natural feather texture, accurate bird anatomy, realistic lighting, believable environment, shallow depth of field, cinematic composition, Ultra HD, 4K.
    """.strip()
