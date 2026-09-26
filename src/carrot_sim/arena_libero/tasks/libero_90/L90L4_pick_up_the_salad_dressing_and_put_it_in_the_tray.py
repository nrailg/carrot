from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.59, -1.91))
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.08, -1.94))
wooden_tray = table_object("wooden_tray", "tray", (2.4, -2.22))
salad_dressing = table_object("salad_dressing", "salad_dressing", (1.98, -2.1))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
    source_class="L90L4PickUpTheSaladDressingAndPutItInTheTray",
    language="Pick up the salad dressing and put it in the tray.",
    objects=(akita_black_bowl, chocolate_pudding, wooden_tray, salad_dressing),
    goal=object_goal(salad_dressing, wooden_tray, region="inside"),
)
