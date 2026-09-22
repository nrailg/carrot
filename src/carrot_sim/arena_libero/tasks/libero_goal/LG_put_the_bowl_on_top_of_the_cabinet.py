from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    fixture_goal,
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
    source_class="LGPutTheBowlOnTopOfTheCabinet",
    suite="libero_goal",
    name="LG_put_the_bowl_on_top_of_the_cabinet",
    language="Put the bowl on the top of the cabinet.",
    objects=(_PLATE, _BOWL, _BOTTLE, _CHEESE, _RACK),
    fixtures=(cabinet(), replace(stove(), position=(3.03, -2.34, 0.782))),
    goal=replace(fixture_goal(_BOWL, "cabinet"), half_size=(0.055, 0.055, 0.30)),
)
