from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.4, -1.96), scale=0.5)
plate = table_object("plate", "plate", (2.4, -2.16), scale=0.8)
porcelain_mug = table_object("porcelain_mug", "white_mug", (2.64, -1.98))
red_coffee_mug = table_object("red_coffee_mug", "red_mug", (2.17, -1.98))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L6_put_the_white_mug_on_the_plate",
    source_class="L90L6PutTheWhiteMugOnThePlate",
    language="Put the white mug on the plate.",
    objects=(chocolate_pudding, plate, porcelain_mug, red_coffee_mug),
    goal=object_goal(porcelain_mug, plate),
)
