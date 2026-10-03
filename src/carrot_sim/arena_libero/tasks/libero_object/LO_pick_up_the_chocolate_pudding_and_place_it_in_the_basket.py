from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

basket = table_object("basket", "basket", (2.65, -2.05))
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.25, -2.2), scale=0.5)
orange_juice = table_object("orange_juice", "orange_juice", (2.58, -2.27))
alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.2, -2.08))
salad_dressing = table_object("salad_dressing", "salad_dressing", (2.44, -1.9))
ketchup_bottle = table_object("ketchup_bottle", "ketchup_bottle", (2.39, -2.08))
bbq_sauce = table_object("bbq_sauce", "bbq_sauce", (2.06, -1.97))

TASK = TaskSpec(
    suite="libero_object",
    name="LO_pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    source_class="LOPickUpTheChocolatePuddingAndPlaceItInTheBasket",
    language="Pick the chocolate pudding and place it in the basket.",
    objects=(
        basket,
        chocolate_pudding,
        orange_juice,
        alphabet_soup,
        salad_dressing,
        ketchup_bottle,
        bbq_sauce,
    ),
    goal=replace(object_goal(chocolate_pudding, basket, region="inside"), release_distance=0.25),
)
