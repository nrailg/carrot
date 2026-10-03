from dataclasses import replace
from math import sqrt

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

basket = table_object("basket", "basket", (2.65, -2.05))
alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.2, -2.08), scale=0.8)
butter = table_object("butter", "butter", (2.08, -1.87), rotation=(0.0, 0.0, sqrt(0.5), sqrt(0.5)))
tomato_sauce = table_object("tomato_sauce", "ketchup", (2.4, -2.25))
cream_cheese = table_object("cream_cheese", "cream_cheese", (2.28, -1.88))
milk = table_object("milk", "milk", (2.05, -2.1))
salad_dressing = table_object("salad_dressing", "salad_dressing", (2.44, -1.9))

TASK = TaskSpec(
    suite="libero_object",
    name="LO_pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    source_class="LOPickUpTheAlphabetSoupAndPlaceItInTheBasket",
    language="Pick the alphabet soup and place it in the basket.",
    objects=(
        basket,
        alphabet_soup,
        butter,
        tomato_sauce,
        cream_cheese,
        milk,
        salad_dressing,
    ),
    goal=replace(object_goal(alphabet_soup, basket, region="inside"), release_distance=0.25),
)
