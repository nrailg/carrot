from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

basket = table_object("basket", "basket", (2.6, -2.26))
alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.2, -2.18), scale=0.8)
butter = table_object("butter", "butter", (2.02, -2.11))
cream_cheese = table_object("cream_cheese", "cream_cheese", (2.03, -1.92))
tomato_sauce = table_object("tomato_sauce", "ketchup", (2.37, -2.25), scale=0.8)

TASK = TaskSpec(
    suite="libero_90",
    name="L90L1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
    source_class="L90L1PickUpTheAlphabetSoupAndPutItInTheBasket",
    language="Pick up the alphabet soup and put it in the basket.",
    objects=(basket, alphabet_soup, butter, cream_cheese, tomato_sauce),
    goal=object_goal(alphabet_soup, basket, region="inside"),
)
