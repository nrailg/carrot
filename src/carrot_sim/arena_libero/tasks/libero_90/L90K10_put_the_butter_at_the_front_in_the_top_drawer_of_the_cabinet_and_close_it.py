from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    drawer_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.42, -1.89))
butter_1 = table_object("butter_1", "butter", (2.2, -1.94))
butter_2 = table_object("butter_2", "butter", (2.4, -2.22))
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.17, -2.18))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
    source_class="L90K10PutTheButterAtTheFrontInTheTopDrawerOfTheCabinetAndCloseIt",
    language="Put the butter at the front in the top drawer of the cabinet and close it.",
    objects=(akita_black_bowl, butter_1, butter_2, chocolate_pudding),
    fixtures=(cabinet(top=0.19),),
    goal=drawer_goal(butter_2, 1, closed=True),
)
