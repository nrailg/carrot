from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    joint_goal,
    table_object,
    winerack,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.41, -1.9), scale=0.7)
wine_bottle = table_object("wine_bottle", "bottle", (2.16, -1.90), scale=0.8)
wine_rack = winerack("wine_rack", (2.15, -2.20))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K4_close_the_bottom_drawer_of_the_cabinet",
    source_class="L90K4CloseTheBottomDrawerOfTheCabinet",
    language="Close the bottom drawer of the cabinet.",
    objects=(akita_black_bowl, wine_bottle, wine_rack),
    fixtures=(cabinet(bottom=0.19),),
    goal=joint_goal("cabinet", "closed", 3),
)
