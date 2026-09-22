from dataclasses import replace

from carrot_sim.arena_libero.tasks.assets import ASSETS
from carrot_sim.arena_libero.tasks.builders import STOVE_SCALE, object_goal, stove, table_object
from carrot_sim.arena_libero.tasks.spec import ObjectSpec, TaskSpec

_STOVE = stove()
_BOWL = ObjectSpec(
    "bowl_target",
    "bowl",
    (
        _STOVE.position[0],
        _STOVE.position[1] - 0.0653,
        # The burner mesh top is 0.013167 m above the unscaled stove origin.
        _STOVE.position[2] + 0.013167 * STOVE_SCALE[2] - ASSETS["bowl"].minimum[2] * 0.6 + 0.003,
    ),
    scale=0.6,
    xy_noise=0.0,
)
_PLATE = table_object("plate", "plate", (2.26, -2.12), scale=0.8)

TASK = TaskSpec(
    source_class="LSPickUpTheBlackBowlOnTheStoveAndPlaceItOnThePlate",
    suite="libero_spatial",
    name="LS_pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    language="Pick up the black bowl on the stove and place it on the plate.",
    objects=(
        _BOWL,
        _PLATE,
        table_object("cookies", "cookies", (2.24, -2.35)),
        table_object("ramekin", "ramekin", (2.22, -1.87), scale=0.5),
        table_object("bowl_distractor", "bowl", (2.65, -1.80), scale=0.6),
    ),
    fixtures=(_STOVE,),
    goal=replace(object_goal(_BOWL, _PLATE), half_size=(0.08, 0.08, 0.15)),
)
