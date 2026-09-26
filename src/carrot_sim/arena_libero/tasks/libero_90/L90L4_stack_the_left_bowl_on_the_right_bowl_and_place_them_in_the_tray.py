from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

wooden_tray = table_object("wooden_tray", "tray", (2.4, -2.22), scale=0.6)
akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.59, -1.91))
akita_black_bowl_right = table_object("akita_black_bowl_right", "bowl", (2.34, -1.91))
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.08, -1.94))
salad_dressing = table_object("salad_dressing", "salad_dressing", (1.98, -2.1))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
    source_class="L90L4StackTheLeftBowlOnTheRightBowlAndPlaceThemInTheTray",
    language="Stack the left bowl on the right bowl and place them in the tray.",
    objects=(
        wooden_tray,
        akita_black_bowl,
        akita_black_bowl_right,
        chocolate_pudding,
        salad_dressing,
    ),
    goal=object_goal(akita_black_bowl, akita_black_bowl_right),
    goals=(object_goal(akita_black_bowl_right, wooden_tray, region="inside"),),
)
