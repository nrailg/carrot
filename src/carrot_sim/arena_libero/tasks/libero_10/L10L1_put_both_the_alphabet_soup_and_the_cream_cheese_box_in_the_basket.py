from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.2, -2.1), scale=0.8)
basket = table_object("basket", "basket", (2.53, -2.13))
cream_cheese = table_object("cream_cheese", "cream_cheese", (2.19, -1.88))
tomato_sauce = table_object("tomato_sauce", "ketchup", (2.25, -2.25), scale=0.8)
ketchup_bottle = table_object("ketchup_bottle", "ketchup_bottle", (2.78, -1.91))

TASK = TaskSpec(
    suite="libero_10",
    name="L10L1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    source_class="L10L1PutBothTheAlphabetSoupAndTheCreamCheeseBoxInTheBasket",
    language="Pick up the alphabet soup and the cream cheese box, and put them in the basket.",
    objects=(alphabet_soup, basket, cream_cheese, tomato_sauce, ketchup_bottle),
    goal=replace(
        object_goal(alphabet_soup, basket, region="inside"),
        release_distance=0.25,
        min_upright_cos=0.0,
    ),
    goals=(
        replace(
            object_goal(cream_cheese, basket, region="inside"),
            release_distance=0.25,
            min_upright_cos=0.0,
        ),
    ),
)
