from dataclasses import replace

from carrot_sim.arena_libero.tasks.assets import ASSETS
from carrot_sim.arena_libero.tasks.builders import CABINET_SCALE, cabinet, object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import ObjectSpec, TaskSpec

_CABINET = cabinet(top=0.18)
_BOWL = ObjectSpec(
    "bowl_target",
    "bowl",
    (
        *_CABINET.position[:2],
        # The cabinet body's unscaled top bound is 0.265601486 m above its origin.
        _CABINET.position[2]
        + 0.265601486 * CABINET_SCALE[2]
        - ASSETS["bowl"].minimum[2] * 0.6
        + 0.003,
    ),
    scale=0.6,
    xy_noise=0.0,
)
_PLATE = table_object("plate", "plate", (2.22, -2.13), scale=0.8)

TASK = TaskSpec(
    source_class="LSPickUpTheBlackBowlOnTheWoodenCabinetAndPlaceItOnThePlate",
    suite="libero_spatial",
    name="LS_pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    language="Pick up the black bowl on the wooden cabinet and place it on the plate.",
    objects=(
        _BOWL,
        _PLATE,
        table_object("cookies", "cookies", (2.23, -2.36)),
        table_object("ramekin", "ramekin", (2.22, -1.87), scale=0.5),
        table_object("bowl_distractor", "bowl", (2.64, -1.79), scale=0.6),
    ),
    fixtures=(_CABINET,),
    goal=replace(object_goal(_BOWL, _PLATE), half_size=(0.08, 0.08, 0.15)),
)
