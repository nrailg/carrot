from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

porcelain_mug = table_object("porcelain_mug", "white_mug", (2.61, -1.98))
red_coffee_mug = table_object("red_coffee_mug", "red_mug", (2.41, -1.98))
white_yellow_mug = table_object("white_yellow_mug", "yellow_white_mug", (2.2, -1.98))
plate = table_object("plate", "plate", (2.23, -2.21), scale=0.8)
plate1 = table_object("plate1", "plate", (2.61, -2.21), scale=0.8)

TASK = TaskSpec(
    suite="libero_90",
    name="L90L5_put_the_yellow_and_white_mug_on_the_right_plate",
    source_class="L90L5PutTheYellowAndWhiteMugOnTheRightPlate",
    language="Put the yellow and white mug on the right plate.",
    objects=(porcelain_mug, red_coffee_mug, white_yellow_mug, plate, plate1),
    goal=object_goal(white_yellow_mug, plate),
)
