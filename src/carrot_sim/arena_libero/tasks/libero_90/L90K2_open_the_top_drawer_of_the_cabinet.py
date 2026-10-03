from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    joint_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl_back = table_object("akita_black_bowl_back", "bowl", (2.32, -1.9), scale=0.8)
plate = table_object("plate", "plate", (2.05, -2.08))
akita_black_bowl_middle = table_object("akita_black_bowl_middle", "bowl", (2.32, -2.1), scale=0.8)
akita_black_bowl_front = table_object("akita_black_bowl_front", "bowl", (2.32, -2.3), scale=0.8)

TASK = TaskSpec(
    suite="libero_90",
    name="L90K2_open_the_top_drawer_of_the_cabinet",
    source_class="L90K2OpenTheTopDrawerOfTheCabinet",
    language="Open the top drawer of the cabinet.",
    objects=(akita_black_bowl_back, plate, akita_black_bowl_middle, akita_black_bowl_front),
    fixtures=(cabinet(),),
    goal=joint_goal("cabinet", "open", 1),
)
