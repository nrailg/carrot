from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.32, -2.1), scale=0.7)
akita_black_bowl_front = table_object("akita_black_bowl_front", "bowl", (2.32, -2.3), scale=0.7)
akita_black_bowl_back = table_object("akita_black_bowl_back", "bowl", (2.32, -1.9), scale=0.7)
plate = table_object("plate", "plate", (2.05, -2.08), scale=0.6)

TASK = TaskSpec(
    suite="libero_90",
    name="L90K2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
    source_class="L90K2StackTheBlackBowlAtTheFrontOnTheBlackBowlInTheMiddle",
    language="Stack the black bowl at the front on the black bowl in the middle.",
    objects=(akita_black_bowl, akita_black_bowl_front, akita_black_bowl_back, plate),
    fixtures=(cabinet(),),
    goal=object_goal(akita_black_bowl_front, akita_black_bowl),
)
