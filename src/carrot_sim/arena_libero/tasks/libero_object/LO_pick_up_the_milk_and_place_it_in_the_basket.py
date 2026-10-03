from dataclasses import replace
from math import sqrt

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

basket = table_object("basket", "basket", (2.65, -2.05))
milk = table_object("milk", "milk", (2.05, -2.1), scale=0.7)
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.25, -2.2))
butter = table_object("butter", "butter", (2.08, -1.87), rotation=(0.0, 0.0, sqrt(0.5), sqrt(0.5)))
orange_juice = table_object("orange_juice", "orange_juice", (2.58, -2.27))
cream_cheese = table_object("cream_cheese", "cream_cheese", (2.28, -1.88))
ketchup_bottle = table_object("ketchup_bottle", "ketchup_bottle", (2.39, -2.08))

TASK = TaskSpec(
    suite="libero_object",
    name="LO_pick_up_the_milk_and_place_it_in_the_basket",
    source_class="LOPickUpTheMilkAndPlaceItInTheBasket",
    language="Pick the milk and place it in the basket.",
    objects=(
        basket,
        milk,
        chocolate_pudding,
        butter,
        orange_juice,
        cream_cheese,
        ketchup_bottle,
    ),
    goal=replace(object_goal(milk, basket, region="inside"), release_distance=0.25),
)
