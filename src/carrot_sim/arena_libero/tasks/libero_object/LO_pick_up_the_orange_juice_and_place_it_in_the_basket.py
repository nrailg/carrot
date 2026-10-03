from dataclasses import replace
from math import sqrt

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

basket = table_object("basket", "basket", (2.65, -2.05))
orange_juice = table_object("orange_juice", "orange_juice", (2.58, -2.27))
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.25, -2.2))
butter = table_object("butter", "butter", (2.08, -1.87), rotation=(0.0, 0.0, sqrt(0.5), sqrt(0.5)))
salad_dressing = table_object("salad_dressing", "salad_dressing", (2.44, -1.9))
ketchup_bottle = table_object("ketchup_bottle", "ketchup_bottle", (2.39, -2.08))
bbq_sauce = table_object("bbq_sauce", "bbq_sauce", (2.06, -1.97))

TASK = TaskSpec(
    suite="libero_object",
    name="LO_pick_up_the_orange_juice_and_place_it_in_the_basket",
    source_class="LOPickUpTheOrangeJuiceAndPlaceItInTheBasket",
    language="Pick the orange juice and place it in the basket.",
    objects=(
        basket,
        orange_juice,
        chocolate_pudding,
        butter,
        salad_dressing,
        ketchup_bottle,
        bbq_sauce,
    ),
    goal=replace(object_goal(orange_juice, basket, region="inside"), release_distance=0.25),
)
