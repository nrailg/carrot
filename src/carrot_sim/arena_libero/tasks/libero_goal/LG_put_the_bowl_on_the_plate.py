from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    object_goal,
    stove,
    table_object,
    winerack,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

_PLATE = table_object("plate", "plate", (2.32, -2.18))
_BOWL = table_object("akita_black_bowl", "bowl", (2.42, -1.89))
_BOTTLE = table_object("wine_bottle", "bottle", (2.16, -1.83), scale=0.8)
_CHEESE = table_object("cream_cheese", "cream_cheese", (2.68, -1.83))
_RACK = winerack("winerack", (1.99, -2.24))

TASK = TaskSpec(
    source_class="LGPutTheBowlOnThePlate",
    suite="libero_goal",
    name="LG_put_the_bowl_on_the_plate",
    language="Pick up the akita black bowl and put it on the plate.",
    objects=(_PLATE, _BOWL, _BOTTLE, _CHEESE, _RACK),
    fixtures=(cabinet(), replace(stove(), position=(3.03, -2.34, 0.782))),
    goal=object_goal(_BOWL, _PLATE),
)
