from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.08, -1.94))
akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.59, -1.91))
wooden_tray = table_object("wooden_tray", "tray", (2.4, -2.22), scale=0.6)
salad_dressing = table_object("salad_dressing", "salad_dressing", (1.98, -2.1))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
    source_class="L90L4PickUpTheChocolatePuddingAndPutItInTheTray",
    language="Pick up the chocolate pudding and put it in the tray.",
    objects=(chocolate_pudding, akita_black_bowl, wooden_tray, salad_dressing),
    goal=object_goal(chocolate_pudding, wooden_tray, region="inside"),
)
