from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    fixture_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

bowl = table_object("bowl", "bowl", (2.42, -1.9))
plate = table_object("plate", "plate", (2.23, -2.14), scale=0.8)

TASK = TaskSpec(
    suite="libero_90",
    name="L90K1_put_the_black_bowl_on_top_of_the_cabinet",
    source_class="L90K1PutTheBlackBowlOnTopOfTheCabinet",
    language="Put the black bowl on top of the cabinet.",
    objects=(bowl, plate),
    fixtures=(cabinet(),),
    goal=fixture_goal(bowl, "cabinet"),
)
