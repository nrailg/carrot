from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.2, -2.18), scale=0.8)
basket = table_object("basket", "basket", (2.6, -2.26))
cream_cheese = table_object("cream_cheese", "cream_cheese", (2.03, -1.92))
tomato_sauce = table_object("tomato_sauce", "ketchup", (2.37, -2.25), scale=0.8)
ketchup_bottle = table_object("ketchup_bottle", "ketchup_bottle", (2.6, -1.94))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L1_pick_up_the_ketchup_and_put_it_in_the_basket",
    source_class="L90L1PickUpTheKetchupAndPutItInTheBasket",
    language="Pick up the ketchup and put it in the basket.",
    objects=(alphabet_soup, basket, cream_cheese, tomato_sauce, ketchup_bottle),
    goal=object_goal(ketchup_bottle, basket, region="inside"),
)
