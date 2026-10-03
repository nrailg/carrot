from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.42, -1.9))
plate = table_object("plate", "plate", (2.23, -2.14))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K1_put_the_black_bowl_on_the_plate",
    source_class="L90K1PutTheBlackBowlOnThePlate",
    language="Put the black bowl on the plate.",
    objects=(akita_black_bowl, plate),
    goal=object_goal(akita_black_bowl, plate),
)
