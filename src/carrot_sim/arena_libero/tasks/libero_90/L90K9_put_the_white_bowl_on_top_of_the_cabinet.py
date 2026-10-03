from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    stove,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

shelf = table_object("shelf", "shelf", (2.2, -2.27), scale=0.8, rotation=(0.0, 0.0, 1.0, 0.0))
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
    name="L90K9_put_the_white_bowl_on_top_of_the_cabinet",
    source_class="L90K9PutTheWhiteBowlOnTopOfTheCabinet",
    language="Put the white bowl on top of the cabinet.",
    objects=(shelf, bowl, frying_pan),
    fixtures=(stove(),),
    goal=object_goal(bowl, shelf),
)
