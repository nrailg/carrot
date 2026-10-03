from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

cream_cheese = table_object("cream_cheese", "cream_cheese", (2.1, -1.95))
wooden_tray = table_object("wooden_tray", "tray", (2.56, -2.24), scale=0.8)
butter = table_object("butter", "butter", (2.04, -2.18))
alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.21, -2.25))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L3_pick_up_the_cream_cheese_and_put_it_in_the_tray",
    source_class="L90L3PickUpTheCreamCheeseAndPutItInTheTray",
    language="Pick up the cream cheese and put it in the tray.",
    objects=(cream_cheese, wooden_tray, butter, alphabet_soup),
    goal=object_goal(cream_cheese, wooden_tray, region="inside"),
)
