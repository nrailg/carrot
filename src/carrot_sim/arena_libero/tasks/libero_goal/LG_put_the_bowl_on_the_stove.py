from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    fixture_goal,
    stove,
    table_object,
    winerack,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

_PLATE = table_object("plate", "plate", (2.26, -2.12), scale=0.8)
_BOWL = table_object("akita_black_bowl", "bowl", (2.44, -1.89), scale=0.8)
_BOTTLE = table_object("wine_bottle", "bottle", (2.18, -1.83), scale=0.8)
_CHEESE = table_object("cream_cheese", "cream_cheese", (2.7, -1.85))
_RACK = winerack("winerack", (1.94, -2.30))

TASK = TaskSpec(
    source_class="LGPutTheBowlOnTheStove",
    suite="libero_goal",
    name="LG_put_the_bowl_on_the_stove",
    language="Put the bowl on the stove.",
    objects=(_PLATE, _BOWL, _BOTTLE, _CHEESE, _RACK),
    fixtures=(replace(cabinet(), position=(3.02, -2.19, 0.9131)), stove()),
    goal=replace(fixture_goal(_BOWL, "stove"), half_size=(0.04, 0.03, 0.30)),
)
