from carrot_sim.arena_libero.tasks.builders import (
    joint_goal,
    microwave,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

plate = table_object("plate", "plate", (2.2, -2.12))
white_bowl = table_object("white_bowl", "white_bowl", (2.47, -1.92))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K7_open_the_microwave",
    source_class="L90K7OpenTheMicrowave",
    language="Open the microwave.",
    objects=(plate, white_bowl),
    fixtures=(microwave(opened=False),),
    goal=joint_goal("microwave", "open"),
)
