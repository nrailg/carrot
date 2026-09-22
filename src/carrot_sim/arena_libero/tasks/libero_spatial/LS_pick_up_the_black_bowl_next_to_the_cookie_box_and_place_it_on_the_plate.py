from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

_BOWL = table_object("bowl_target", "bowl", (2.40, -2.06), scale=0.6)
_PLATE = table_object("plate", "plate", (2.66, -2.27), scale=0.8)

TASK = TaskSpec(
    source_class="LSPickUpTheBlackBowlNextToTheCookieBoxAndPlaceItOnThePlate",
    suite="libero_spatial",
    name="LS_pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    language="Pick up the black bowl next to the cookie box and place it on the plate.",
    objects=(
        _BOWL,
        _PLATE,
        table_object("cookies", "cookies", (2.23, -2.06)),
        table_object("ramekin", "ramekin", (2.24, -2.31), scale=0.5),
        table_object("bowl_distractor", "bowl", (2.67, -1.84), scale=0.6),
    ),
    goal=replace(object_goal(_BOWL, _PLATE), half_size=(0.08, 0.08, 0.15)),
)
