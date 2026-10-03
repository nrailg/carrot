from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    object_goal,
    stove,
    table_object,
    winerack,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

_PLATE = table_object("plate", "plate", (2.62, -1.94))
_BOWL = table_object("akita_black_bowl", "bowl", (2.48, -2.29))
_BOTTLE = table_object("wine_bottle", "bottle", (2.40, -1.83), scale=0.8)
_CHEESE = table_object("cream_cheese", "cream_cheese", (2.68, -2.22))
_RACK = winerack("winerack", (2.2, -2.1))

TASK = TaskSpec(
    source_class="LGPutTheWineBottleOnTheRack",
    suite="libero_goal",
    name="LG_put_the_wine_bottle_on_the_rack",
    language="Put the wine bottle on the rack.",
    objects=(_PLATE, _BOWL, _BOTTLE, _CHEESE, _RACK),
    fixtures=(
        replace(cabinet(), position=(2.97, -2.37, 0.9131)),
        replace(stove(), position=(1.865, -2.33, 0.782)),
    ),
    goal=replace(object_goal(_BOTTLE, _RACK), half_size=(0.14, 0.09, 0.20)),
)
