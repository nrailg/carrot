from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    drawer_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

plate = table_object("plate", "plate", (2.22, -2.14))
akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.42, -1.9))
ketchup = table_object("ketchup", "ketchup", (2.43, -2.26))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
    source_class="L90K5PutTheKetchupInTheTopDrawerOfTheCabinet",
    language="Put the ketchup in the top drawer of the cabinet.",
    objects=(plate, akita_black_bowl, ketchup),
    fixtures=(cabinet(top=0.19),),
    goal=drawer_goal(ketchup, 1),
)
