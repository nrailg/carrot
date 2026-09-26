from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

basket = table_object("basket", "basket", (2.6, -2.26))
alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.2, -2.18))
butter = table_object("butter", "butter", (2.02, -2.11))
milk_drink = table_object("milk_drink", "milk", (2.43, -1.94))
orange_juice = table_object("orange_juice", "orange_juice", (2.24, -1.94))
tomato_sauce = table_object("tomato_sauce", "ketchup", (2.37, -2.25), scale=0.8)
ketchup_bottle = table_object("ketchup_bottle", "ketchup_bottle", (2.6, -1.94))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L2_pick_up_the_butter_and_put_it_in_the_basket",
    source_class="L90L2PickUpTheButterAndPutItInTheBasket",
    language="Pick up the butter and put it in the basket.",
    objects=(basket, alphabet_soup, butter, milk_drink, orange_juice, tomato_sauce, ketchup_bottle),
    goal=object_goal(butter, basket, region="inside"),
)
