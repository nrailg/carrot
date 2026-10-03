from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    joint_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.42, -1.9))
plate = table_object("plate", "plate", (2.23, -2.14))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K1_open_the_bottom_drawer_of_the_cabinet",
    source_class="L90K1OpenTheBottomDrawerOfTheCabinet",
    language="Open the bottom drawer of the cabinet.",
    objects=(akita_black_bowl, plate),
    fixtures=(cabinet(),),
    goal=joint_goal("cabinet", "open", 3),
)
