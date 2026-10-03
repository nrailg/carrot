from carrot_sim.arena_libero.tasks.spec import GoalSpec, ObjectSpec, TaskSpec

TASK = TaskSpec(
    source_class="LSPickUpTheBlackBowlFromTableCenterAndPlaceItOnThePlate",
    suite="libero_spatial",
    name="LS_pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    language="Pick up the black bowl from table center and place it on the plate.",
    objects=(
        ObjectSpec("bowl_target", "bowl", (2.439, -2.184, 0.805)),
        ObjectSpec("plate", "plate", (2.67, -2.25, 0.775), scale=0.8),
        ObjectSpec("ramekin", "ramekin", (2.68, -1.77, 0.80), scale=0.5),
        ObjectSpec("bowl_distractor", "bowl", (2.69, -2.0, 0.805)),
        ObjectSpec("cookies", "cookies", (2.27, -2.28, 0.80)),
    ),
    goal=GoalSpec("on_plate", "bowl_target", "plate", half_size=(0.1, 0.1, 0.15)),
)
