from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

plate = table_object("plate", "plate", (2.22, -2.14))
akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.42, -1.9))
ketchup = table_object("ketchup", "ketchup", (2.43, -2.26))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K5_put_the_black_bowl_on_the_plate",
    source_class="L90K5PutTheBlackBowlOnThePlate",
    language="Put the black bowl on the plate.",
    objects=(plate, akita_black_bowl, ketchup),
    fixtures=(cabinet(top=0.19),),
    goal=object_goal(akita_black_bowl, plate),
)
