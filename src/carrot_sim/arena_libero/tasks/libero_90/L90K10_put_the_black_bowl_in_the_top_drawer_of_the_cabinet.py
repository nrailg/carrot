from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    drawer_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.42, -1.89), scale=0.7)
butter_1 = table_object("butter_1", "butter", (2.2, -1.94))
butter_2 = table_object("butter_2", "butter", (2.4, -2.22))
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.17, -2.18))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
    source_class="L90K10PutTheBlackBowlInTheTopDrawerOfTheCabinet",
    language="Put the black bowl in the top drawer of the cabinet.",
    objects=(akita_black_bowl, butter_1, butter_2, chocolate_pudding),
    fixtures=(cabinet(top=0.19),),
    goal=drawer_goal(akita_black_bowl, 1),
)
