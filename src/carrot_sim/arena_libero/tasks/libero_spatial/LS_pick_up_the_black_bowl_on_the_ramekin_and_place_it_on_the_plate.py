from dataclasses import replace

from carrot_sim.arena_libero.tasks.assets import ASSETS
from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import ObjectSpec, TaskSpec

_RAMEKIN = table_object("ramekin", "ramekin", (2.36, -2.12), scale=0.5, xy_noise=0.0)
_BOWL = ObjectSpec(
    "bowl_target",
    "bowl",
    (
        *_RAMEKIN.position[:2],
        _RAMEKIN.position[2]
        + ASSETS["ramekin"].maximum[2] * 0.5
        - ASSETS["bowl"].minimum[2] * 0.6
        + 0.003,
    ),
    scale=0.6,
    xy_noise=0.0,
)
_PLATE = table_object("plate", "plate", (2.65, -2.18), scale=0.8)

TASK = TaskSpec(
    source_class="LSPickUpTheBlackBowlOnTheRamekinAndPlaceItOnThePlate",
    suite="libero_spatial",
    name="LS_pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    language="Pick up the black bowl on the ramekin and place it on the plate.",
    objects=(
        _BOWL,
        _PLATE,
        table_object("cookies", "cookies", (2.22, -2.35)),
        _RAMEKIN,
        table_object("bowl_distractor", "bowl", (2.62, -1.85), scale=0.6),
    ),
    goal=replace(object_goal(_BOWL, _PLATE), half_size=(0.08, 0.08, 0.15)),
)
