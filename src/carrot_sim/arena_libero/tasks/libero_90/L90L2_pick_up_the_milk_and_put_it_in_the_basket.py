from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

basket = table_object("basket", "basket", (2.6, -2.26))
milk_drink = table_object("milk_drink", "milk", (2.43, -1.94), scale=0.8)
alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.2, -2.18))
ketchup = table_object("ketchup", "ketchup", (2.37, -2.25))
orange_juice = table_object("orange_juice", "orange_juice", (2.24, -1.94))
cream_cheese_stick = table_object("cream_cheese_stick", "cream_cheese", (2.03, -1.92))
ketchup_bottle = table_object("ketchup_bottle", "ketchup_bottle", (2.6, -1.94))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L2_pick_up_the_milk_and_put_it_in_the_basket",
    source_class="L90L2PickUpTheMilkAndPutItInTheBasket",
    language="Pick up the milk and put it in the basket.",
    objects=(
        basket,
        milk_drink,
        alphabet_soup,
        ketchup,
        orange_juice,
        cream_cheese_stick,
        ketchup_bottle,
    ),
    goal=object_goal(milk_drink, basket, region="inside"),
)
