from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.21, -2.25), scale=0.8)
wooden_tray = table_object("wooden_tray", "tray", (2.56, -2.24), scale=0.6)
butter = table_object("butter", "butter", (2.04, -2.18))
cream_cheese = table_object("cream_cheese", "cream_cheese", (2.1, -1.95))
tomato_sauce = table_object("tomato_sauce", "ketchup", (2.28, -1.95), scale=0.8)
ketchup_bottle = table_object("ketchup_bottle", "ketchup_bottle", (2.5, -1.94))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L3_pick_up_the_ketchup_and_put_it_in_the_tray",
    source_class="L90L3PickUpTheKetchupAndPutItInTheTray",
    language="Pick up the ketchup and put it in the tray.",
    objects=(alphabet_soup, wooden_tray, butter, cream_cheese, tomato_sauce, ketchup_bottle),
    goal=object_goal(ketchup_bottle, wooden_tray, region="inside"),
)
