from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.2, -2.1), scale=0.8)
basket = table_object("basket", "basket", (2.53, -2.13))
butter = table_object("butter", "butter", (2.1, -2.07))
cream_cheese = table_object("cream_cheese", "cream_cheese", (2.19, -1.88))
milk = table_object("milk", "milk", (2.64, -1.875))
orange_juice = table_object("orange_juice", "orange_juice", (2.45, -1.88))
tomato_sauce = table_object("tomato_sauce", "ketchup", (2.25, -2.25), scale=0.8)
ketchup_bottle = table_object("ketchup_bottle", "ketchup_bottle", (2.78, -1.91))

TASK = TaskSpec(
    suite="libero_10",
    name="L10L2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    source_class="L10L2PutBothTheCreamCheeseBoxAndTheButterInTheBasket",
    language="Pick up the cream cheese box and the butter, and put them in the basket.",
    objects=(
        alphabet_soup,
        basket,
        butter,
        cream_cheese,
        milk,
        orange_juice,
        tomato_sauce,
        ketchup_bottle,
    ),
    goal=replace(
        object_goal(cream_cheese, basket, region="inside"),
        release_distance=0.25,
        min_upright_cos=0.0,
    ),
    goals=(
        replace(
            object_goal(butter, basket, region="inside"),
            release_distance=0.25,
            min_upright_cos=0.0,
        ),
    ),
)
