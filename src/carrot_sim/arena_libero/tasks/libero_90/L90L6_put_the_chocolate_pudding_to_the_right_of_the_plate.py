from carrot_sim.arena_libero.tasks.builders import (
    beside_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

plate = table_object("plate", "plate", (2.4, -2.16))
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.4, -1.96))
porcelain_mug = table_object("porcelain_mug", "white_mug", (2.64, -1.98))
red_coffee_mug = table_object("red_coffee_mug", "red_mug", (2.17, -1.98))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L6_put_the_chocolate_pudding_to_the_right_of_the_plate",
    source_class="L90L6PutTheChocolatePuddingToTheRightOfThePlate",
    language="Put the chocolate pudding to the right of the plate.",
    objects=(plate, chocolate_pudding, porcelain_mug, red_coffee_mug),
    goal=beside_goal(chocolate_pudding, plate, "right"),
)
