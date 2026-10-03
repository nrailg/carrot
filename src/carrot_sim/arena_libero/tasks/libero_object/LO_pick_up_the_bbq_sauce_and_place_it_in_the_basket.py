from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

basket = table_object("basket", "basket", (2.65, -2.05))
alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.2, -2.08))
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.25, -2.2))
cream_cheese = table_object("cream_cheese", "cream_cheese", (2.28, -1.88))
salad_dressing = table_object("salad_dressing", "salad_dressing", (2.44, -1.9))
ketchup_bottle = table_object("ketchup_bottle", "ketchup_bottle", (2.39, -2.08))
bbq_sauce = table_object("bbq_sauce", "bbq_sauce", (2.06, -1.97))

TASK = TaskSpec(
    suite="libero_object",
    name="LO_pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    source_class="LOPickUpTheBbqSauceAndPlaceItInTheBasket",
    language="Pick the bbq sauce and place it in the basket.",
    objects=(
        basket,
        alphabet_soup,
        chocolate_pudding,
        cream_cheese,
        salad_dressing,
        ketchup_bottle,
        bbq_sauce,
    ),
    goal=replace(object_goal(bbq_sauce, basket, region="inside"), release_distance=0.25),
)
