from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    joint_goal,
    table_object,
    winerack,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.42, -1.89))
butter = table_object("butter", "butter", (2.2, -1.94))
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.17, -2.18))
wine_rack = winerack("wine_rack", (1.88, -2.37))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K10_close_the_top_drawer_of_the_cabinet",
    source_class="L90K10CloseTheTopDrawerOfTheCabinet",
    language="Close the top drawer of the cabinet.",
    objects=(akita_black_bowl, butter, chocolate_pudding, wine_rack),
    fixtures=(cabinet(top=0.19),),
    goal=joint_goal("cabinet", "closed", 1),
)
