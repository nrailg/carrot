from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl_back = table_object("akita_black_bowl_back", "bowl", (2.32, -1.9), scale=0.8)
plate = table_object("plate", "plate", (2.05, -2.08))
akita_black_bowl_middle = table_object("akita_black_bowl_middle", "bowl", (2.32, -2.1), scale=0.8)
akita_black_bowl_front = table_object("akita_black_bowl_front", "bowl", (2.32, -2.3), scale=0.8)

TASK = TaskSpec(
    suite="libero_90",
    name="L90K2_put_the_black_bowl_at_the_front_on_the_plate",
    source_class="L90K2PutTheBlackBowlAtTheFrontOnThePlate",
    language="Put the black bowl at the front on the plate.",
    objects=(akita_black_bowl_back, plate, akita_black_bowl_middle, akita_black_bowl_front),
    fixtures=(cabinet(),),
    goal=object_goal(akita_black_bowl_front, plate),
)
