from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    drawer_goal,
    joint_goal,
    table_object,
    winerack,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.42, -1.9), scale=0.7)
plate = table_object("plate", "plate", (2.23, -2.14))
wine_rack = winerack("wine_rack", (1.88, -2.37))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
    source_class="L90K1OpenTheTopDrawerOfTheCabinetAndPutTheBowlInIt",
    language="Open the top drawer of the cabinet and put the bowl in it.",
    objects=(akita_black_bowl, plate, wine_rack),
    fixtures=(cabinet(),),
    goal=drawer_goal(akita_black_bowl, 1),
    goals=(joint_goal("cabinet", "open", 1),),
)
