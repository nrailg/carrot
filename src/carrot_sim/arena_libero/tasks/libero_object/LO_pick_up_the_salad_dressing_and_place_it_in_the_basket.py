from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

basket = table_object("basket", "basket", (2.65, -2.05))
alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.2, -2.08))
cream_cheese = table_object("cream_cheese", "cream_cheese", (2.28, -1.88))
milk = table_object("milk", "milk", (2.05, -2.1))
tomato_sauce = table_object("tomato_sauce", "ketchup", (2.4, -2.25))
ketchup_bottle = table_object("ketchup_bottle", "ketchup_bottle", (2.39, -2.08))
salad_dressing = table_object("salad_dressing", "salad_dressing", (2.44, -1.9))

TASK = TaskSpec(
    suite="libero_object",
    name="LO_pick_up_the_salad_dressing_and_place_it_in_the_basket",
    source_class="LOPickUpTheSaladDressingAndPlaceItInTheBasket",
    language="Pick the salad dressing and place it in the basket.",
    objects=(
        basket,
        alphabet_soup,
        cream_cheese,
        milk,
        tomato_sauce,
        ketchup_bottle,
        salad_dressing,
    ),
    goal=replace(object_goal(salad_dressing, basket, region="inside"), release_distance=0.35),
)
