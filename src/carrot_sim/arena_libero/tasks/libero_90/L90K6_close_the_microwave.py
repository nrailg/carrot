from carrot_sim.arena_libero.tasks.builders import (
    joint_goal,
    microwave,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

porcelain_mug = table_object("porcelain_mug", "white_mug", (2.18, -2.1))
white_yellow_mug = table_object("white_yellow_mug", "yellow_white_mug", (2.42, -1.91))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K6_close_the_microwave",
    source_class="L90K6CloseTheMicrowave",
    language="Close the microwave.",
    objects=(porcelain_mug, white_yellow_mug),
    fixtures=(microwave(opened=True),),
    goal=joint_goal("microwave", "closed"),
)
