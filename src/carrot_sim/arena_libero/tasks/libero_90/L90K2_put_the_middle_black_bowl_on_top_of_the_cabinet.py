from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    fixture_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.32, -1.9))
akita_black_bowl1 = table_object("akita_black_bowl1", "bowl", (2.32, -2.1))
akita_black_bowl2 = table_object("akita_black_bowl2", "bowl", (2.32, -2.3))
plate = table_object("plate", "plate", (2.05, -2.08), scale=0.6)

TASK = TaskSpec(
    suite="libero_90",
    name="L90K2_put_the_middle_black_bowl_on_top_of_the_cabinet",
    source_class="L90K2PutTheMiddleBlackBowlOnTopOfTheCabinet",
    language="Put the middle black bowl on top of the cabinet.",
    objects=(akita_black_bowl, akita_black_bowl1, akita_black_bowl2, plate),
    fixtures=(cabinet(),),
    goal=fixture_goal(akita_black_bowl1, "cabinet"),
)
