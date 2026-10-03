from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    drawer_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

butter = table_object("butter", "butter", (2.2, -1.94), scale=0.8)
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.17, -2.18))
akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.42, -1.89))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
    source_class="L90K10PutTheButterAtTheBackInTheTopDrawerOfTheCabinetAndCloseIt",
    language="Put the butter at the back in the top drawer of the cabinet and close it.",
    objects=(butter, chocolate_pudding, akita_black_bowl),
    fixtures=(cabinet(top=0.19),),
    goal=drawer_goal(butter, 1, closed=True),
)
