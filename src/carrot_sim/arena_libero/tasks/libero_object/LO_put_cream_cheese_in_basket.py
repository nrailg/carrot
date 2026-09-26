from dataclasses import replace
from math import sqrt

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

basket = table_object("basket", "basket", (2.65, -2.05))
alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.2, -2.08))
butter = table_object("butter", "butter", (2.08, -1.87), rotation=(0.0, 0.0, sqrt(0.5), sqrt(0.5)))
milk = table_object("milk", "milk", (2.05, -2.1))
orange_juice = table_object("orange_juice", "orange_juice", (2.58, -2.27))
tomato_sauce = table_object("tomato_sauce", "ketchup", (2.4, -2.25))
cream_cheese = table_object("cream_cheese", "cream_cheese", (2.28, -1.88))

TASK = TaskSpec(
    suite="libero_object",
    name="LO_put_cream_cheese_in_basket",
    source_class="LOPutCreamCheeseInBasket",
    language="Pick the cream cheese and place it in the basket.",
    objects=(
        basket,
        alphabet_soup,
        butter,
        milk,
        orange_juice,
        tomato_sauce,
        cream_cheese,
    ),
    goal=replace(object_goal(cream_cheese, basket, region="inside"), release_distance=0.25),
)
