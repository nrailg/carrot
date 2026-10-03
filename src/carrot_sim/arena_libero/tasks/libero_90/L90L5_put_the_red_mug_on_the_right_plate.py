from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

white_yellow_mug = table_object("white_yellow_mug", "yellow_white_mug", (2.2, -1.98))
plate = table_object("plate", "plate", (2.23, -2.21))
porcelain_mug = table_object("porcelain_mug", "white_mug", (2.61, -1.98))
red_coffee_mug = table_object("red_coffee_mug", "red_mug", (2.41, -1.98))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L5_put_the_red_mug_on_the_right_plate",
    source_class="L90L5PutTheRedMugOnTheRightPlate",
    language="Put the red mug on the right plate.",
    objects=(white_yellow_mug, plate, porcelain_mug, red_coffee_mug),
    goal=object_goal(red_coffee_mug, plate),
)
