from dataclasses import replace
from math import sqrt

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

basket = table_object("basket", "basket", (2.65, -2.05))
tomato_sauce = table_object("tomato_sauce", "ketchup", (2.4, -2.25))
butter = table_object("butter", "butter", (2.08, -1.87), rotation=(0.0, 0.0, sqrt(0.5), sqrt(0.5)))
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.25, -2.2))
orange_juice = table_object("orange_juice", "orange_juice", (2.58, -2.27))
milk = table_object("milk", "milk", (2.05, -2.1))
bbq_sauce = table_object("bbq_sauce", "bbq_sauce", (2.06, -1.97))

TASK = TaskSpec(
    suite="libero_object",
    name="LO_pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    source_class="LOPickUpTheTomatoSauceAndPlaceItInTheBasket",
    language="Pick the tomato sauce and place it in the basket.",
    objects=(
        basket,
        tomato_sauce,
        butter,
        chocolate_pudding,
        orange_juice,
        milk,
        bbq_sauce,
    ),
    goal=replace(object_goal(tomato_sauce, basket, region="inside"), release_distance=0.25),
)
