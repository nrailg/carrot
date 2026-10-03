from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

tray = table_object("tray", "tray", (2.56, -2.24), scale=0.6)
alphabet_soup = table_object("alphabet_soup", "alphabet_soup", (2.21, -2.25), scale=0.8)
butter = table_object("butter", "butter", (2.04, -2.18))
ketchup = table_object("ketchup", "ketchup", (2.28, -1.95))
cream_cheese_stick = table_object("cream_cheese_stick", "cream_cheese", (2.1, -1.95))
ketchup_bottle = table_object("ketchup_bottle", "ketchup_bottle", (2.5, -1.94))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L3_pick_up_the_butter_and_put_it_in_the_tray",
    source_class="L90L3PickUpTheButterAndPutItInTheTray",
    language="Pick up the butter and put it in the tray.",
    objects=(tray, alphabet_soup, butter, ketchup, cream_cheese_stick, ketchup_bottle),
    goal=object_goal(butter, tray, region="inside"),
)
