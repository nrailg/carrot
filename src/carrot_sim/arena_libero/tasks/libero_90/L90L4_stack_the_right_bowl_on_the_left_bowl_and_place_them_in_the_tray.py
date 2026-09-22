from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

tray = table_object("tray", "tray", (2.4, -2.22), scale=0.6)
chocolate_pudding = table_object("chocolate_pudding", "chocolate_pudding", (2.08, -1.94), scale=0.5)
bowl_left = table_object("bowl_left", "bowl", (2.59, -1.91), scale=0.6)
bowl_right = table_object("bowl_right", "bowl", (2.34, -1.91), scale=0.6)
salad_dressing = table_object("salad_dressing", "salad_dressing", (1.98, -2.1))

TASK = TaskSpec(
    suite="libero_90",
    name="L90L4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
    source_class="L90L4StackTheRightBowlOnTheLeftBowlAndPlaceThemInTheTray",
    language="Stack the right bowl on the left bowl and place them in the tray.",
    objects=(tray, chocolate_pudding, bowl_left, bowl_right, salad_dressing),
    goal=object_goal(bowl_right, bowl_left),
    goals=(object_goal(bowl_left, tray, region="inside"),),
)
