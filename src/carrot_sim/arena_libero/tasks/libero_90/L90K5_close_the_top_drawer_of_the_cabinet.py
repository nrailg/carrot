from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    joint_goal,
    table_object,
    winerack,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.42, -1.9))
ketchup = table_object("ketchup", "ketchup", (2.43, -2.26))
plate = table_object("plate", "plate", (2.22, -2.14))
wine_rack = winerack("wine_rack", (1.88, -2.37))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K5_close_the_top_drawer_of_the_cabinet",
    source_class="L90K5CloseTheTopDrawerOfTheCabinet",
    language="Close the top drawer of the cabinet.",
    objects=(akita_black_bowl, ketchup, plate, wine_rack),
    fixtures=(cabinet(top=0.19),),
    goal=joint_goal("cabinet", "closed", 1),
)
