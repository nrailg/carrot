from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    drawer_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.42, -1.9), scale=0.8)
ketchup = table_object("ketchup", "ketchup", (2.43, -2.26))
plate = table_object("plate", "plate", (2.22, -2.14))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
    source_class="L90K5PutTheBlackBowlInTheTopDrawerOfTheCabinet",
    language="Put the black bowl in the top drawer of the cabinet.",
    objects=(akita_black_bowl, ketchup, plate),
    fixtures=(cabinet(top=0.19),),
    goal=drawer_goal(akita_black_bowl, 1),
)
