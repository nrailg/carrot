from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

plate = table_object("plate", "plate", (2.23, -2.14), scale=0.8)
plate_left = table_object("plate_left", "plate", (2.61, -2.14), scale=0.8)
porcelain_mug = table_object("porcelain_mug", "white_mug", (2.61, -1.93))
red_coffee_mug = table_object("red_coffee_mug", "red_mug", (2.41, -1.93))
white_yellow_mug = table_object("white_yellow_mug", "yellow_white_mug", (2.2, -1.93))

TASK = TaskSpec(
    suite="libero_10",
    name="L10L5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    source_class="L10L5PutWhiteMugOnLeftPlateAndPutYellowAndWhiteMugOnRightPlate",
    language=(
        "Put the white mug on the left plate and put the yellow and white mug on the right plate."
    ),
    objects=(plate, plate_left, porcelain_mug, red_coffee_mug, white_yellow_mug),
    goal=replace(object_goal(porcelain_mug, plate_left), release_distance=0.25),
    goals=(replace(object_goal(white_yellow_mug, plate), release_distance=0.25),),
)
