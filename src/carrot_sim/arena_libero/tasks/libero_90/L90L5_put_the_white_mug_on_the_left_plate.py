from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

plate = table_object("plate", "plate", (2.23, -2.21), scale=0.8)
plate_left = table_object("plate_left", "plate", (2.61, -2.21), scale=0.8)
porcelain_mug = table_object("porcelain_mug", "white_mug", (2.61, -1.98))
red_coffee_mug = table_object("red_coffee_mug", "red_mug", (2.41, -1.98))
white_yellow_mug = table_object("white_yellow_mug", "yellow_white_mug", (2.2, -1.98))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L5_put_the_white_mug_on_the_left_plate",
    source_class="L90L5PutTheWhiteMugOnTheLeftPlate",
    language="Put the white mug on the left plate.",
    objects=(plate, plate_left, porcelain_mug, red_coffee_mug, white_yellow_mug),
    goal=object_goal(porcelain_mug, plate_left),
)
