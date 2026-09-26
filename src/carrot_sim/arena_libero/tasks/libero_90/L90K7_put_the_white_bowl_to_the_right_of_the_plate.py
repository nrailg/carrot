from carrot_sim.arena_libero.tasks.builders import (
    beside_goal,
    microwave,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

plate = table_object("plate", "plate", (2.2, -2.12))
white_bowl = table_object("white_bowl", "white_bowl", (2.47, -1.92))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K7_put_the_white_bowl_to_the_right_of_the_plate",
    source_class="L90K7PutTheWhiteBowlToTheRightOfThePlate",
    language="Put the white bowl to the right of the plate.",
    objects=(plate, white_bowl),
    fixtures=(microwave(opened=False),),
    goal=beside_goal(white_bowl, plate, "right"),
)
