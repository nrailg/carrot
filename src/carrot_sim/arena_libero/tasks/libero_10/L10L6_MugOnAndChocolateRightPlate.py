from dataclasses import replace
from math import sqrt

from carrot_sim.arena_libero.tasks.builders import beside_goal, object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

chocolate_pudding = table_object(
    "chocolate_pudding",
    "chocolate_pudding",
    (2.42, -1.88),
    rotation=(0.0, 0.0, sqrt(0.5), sqrt(0.5)),
)
plate = table_object("plate", "plate", (2.45, -2.16))
porcelain_mug = table_object("porcelain_mug", "white_mug", (2.68, -1.93))
red_coffee_mug = table_object("red_coffee_mug", "red_mug", (2.17, -1.95))

TASK = TaskSpec(
    suite="libero_10",
    name="L10L6_MugOnAndChocolateRightPlate",
    source_class="L10L6PutWhiteMugOnPlateAndPutChocolatePuddingToRightPlate",
    language=(
        "Pick up the white mug and put it on the plate, "
        "and put the chocolate pudding to the right of the plate."
    ),
    objects=(chocolate_pudding, plate, porcelain_mug, red_coffee_mug),
    goal=replace(object_goal(porcelain_mug, plate), release_distance=0.35),
    goals=(
        replace(
            beside_goal(chocolate_pudding, plate, "right"),
            release_distance=0.35,
            min_upright_cos=0.95,
            side_tolerance=0.4,
        ),
    ),
)
