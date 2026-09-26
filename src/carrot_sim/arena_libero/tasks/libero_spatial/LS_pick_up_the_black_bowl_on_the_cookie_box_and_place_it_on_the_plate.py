from dataclasses import replace

from carrot_sim.arena_libero.tasks.assets import ASSETS
from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import ObjectSpec, TaskSpec

_COOKIES = table_object("cookies", "cookies", (2.36, -2.12), xy_noise=0.0)
_BOWL = ObjectSpec(
    "bowl_target",
    "bowl",
    (
        *_COOKIES.position[:2],
        _COOKIES.position[2]
        + ASSETS["cookies"].maximum[2]
        - ASSETS["bowl"].minimum[2] * 0.6
        + 0.003,
    ),
    scale=0.6,
    xy_noise=0.0,
)
_PLATE = table_object("plate", "plate", (2.65, -2.18), scale=0.8)

TASK = TaskSpec(
    source_class="LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate",
    suite="libero_spatial",
    name="LS_pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    language="Pick up the black bowl on the cookie box and place it on the plate.",
    objects=(
        _BOWL,
        _PLATE,
        _COOKIES,
        table_object("ramekin", "ramekin", (2.24, -1.85), scale=0.5),
        table_object("bowl_distractor", "bowl", (2.62, -1.85), scale=0.6),
    ),
    goal=replace(object_goal(_BOWL, _PLATE), half_size=(0.08, 0.08, 0.15)),
)
