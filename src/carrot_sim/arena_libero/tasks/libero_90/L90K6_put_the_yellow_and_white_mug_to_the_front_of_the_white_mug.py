from carrot_sim.arena_libero.tasks.builders import (
    beside_goal,
    microwave,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

porcelain_mug = table_object("porcelain_mug", "white_mug", (2.18, -2.1))
white_yellow_mug = table_object("white_yellow_mug", "yellow_white_mug", (2.42, -1.91))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
    source_class="L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug",
    language="Put the yellow and white mug to the front of the white mug.",
    objects=(porcelain_mug, white_yellow_mug),
    fixtures=(microwave(opened=True),),
    goal=beside_goal(white_yellow_mug, porcelain_mug, "front"),
)
