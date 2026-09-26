from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    stove,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

shelf = table_object("shelf", "shelf", (2.2, -2.27), scale=1.2, rotation=(0.0, 0.0, 1.0, 0.0))
bowl = table_object("bowl", "ramekin", (2.81, -1.91))
frying_pan = table_object(
    "frying_pan",
    "frying_pan",
    (2.12, -1.96),
    scale=0.8,
    rotation=(0.0, 0.0, -0.7071067811865476, 0.7071067811865476),
)

TASK = TaskSpec(
    suite="libero_90",
    name="L90K9_put_the_frying_pan_on_the_cabinet_shelf",
    source_class="L90K9PutTheFryingPanOnTheCabinetShelf",
    language="Put the frying pan on the cabinet shelf.",
    objects=(shelf, bowl, frying_pan),
    fixtures=(stove(),),
    goal=replace(object_goal(frying_pan, shelf, region="middle"), containment="partial"),
)
