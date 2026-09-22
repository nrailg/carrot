from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    scaled_bounds,
    stove,
    table_object,
    winerack,
)
from carrot_sim.arena_libero.tasks.spec import GoalSpec, TaskSpec

_PLATE = table_object("plate", "plate", (2.28, -2.16))
_BOWL = table_object("akita_black_bowl", "bowl", (2.20, -1.89))
_BOTTLE = table_object("wine_bottle", "bottle", (2.39, -1.83), scale=0.8)
_CHEESE = table_object("cream_cheese", "cream_cheese", (2.25, -2.39))
_RACK = winerack("winerack", (1.86, -2.31))

TASK = TaskSpec(
    source_class="LGPushThePlateToTheFrontOfTheStove",
    suite="libero_goal",
    name="LG_push_the_plate_to_the_front_of_the_stove",
    language="Push the plate to the front of the stove.",
    objects=(_PLATE, _BOWL, _BOTTLE, _CHEESE, _RACK),
    fixtures=(
        replace(cabinet(), position=(3.02, -2.30, 0.9131)),
        replace(stove(), position=(2.60, -1.92, 0.782)),
    ),
    goal=GoalSpec(
        "relative",
        _PLATE.name,
        "stove",
        support_body="Stovetop031",
        # Source front is world -y; the region also bounds both lateral directions.
        frame="world",
        center=(0.0, -0.38, -0.0113915),
        half_size=(0.02, 0.005, 0.02),
        target_center=scaled_bounds(_PLATE)[0],
        target_half_size=scaled_bounds(_PLATE)[1],
        release_distance=0.35,
    ),
)
