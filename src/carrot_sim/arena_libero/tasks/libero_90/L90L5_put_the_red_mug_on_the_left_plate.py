from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

plate_left = table_object("plate_left", "plate", (2.61, -2.21))
plate_right = table_object("plate_right", "plate", (2.23, -2.21))
red_coffee_mug = table_object("red_coffee_mug", "red_mug", (2.41, -1.98))
porcelain_mug = table_object("porcelain_mug", "white_mug", (2.61, -1.98))
white_yellow_mug = table_object("white_yellow_mug", "yellow_white_mug", (2.2, -1.98))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L5_put_the_red_mug_on_the_left_plate",
    source_class="L90L5PutTheRedMugOnTheLeftPlate",
    language="Put the red mug on the left plate.",
    objects=(plate_left, plate_right, red_coffee_mug, porcelain_mug, white_yellow_mug),
    goal=object_goal(red_coffee_mug, plate_left),
)
