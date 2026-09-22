from carrot_sim.arena_libero.tasks.spec import GoalSpec, ObjectSpec, TaskSpec

TASK = TaskSpec(
    suite="libero_spatial",
    name="LS_pick_up_black_bowl_between_plate_and_ramekin_and_place_it_on_plate",
    language="Pick up the black bowl between the plate and the ramekin and place it on the plate.",
    objects=(
        ObjectSpec("bowl_target", "bowl", (2.42, -2.06, 0.787), scale=0.6),
        ObjectSpec("plate", "plate", (2.62, -2.06, 0.775), scale=0.8),
        ObjectSpec("ramekin", "ramekin", (2.22, -2.06, 0.80), scale=0.5),
        ObjectSpec("bowl_distractor", "bowl", (2.64, -1.82, 0.787), scale=0.6),
        ObjectSpec("cookies", "cookies", (2.26, -2.28, 0.80)),
    ),
    goal=GoalSpec("on_plate", "bowl_target", "plate", half_size=(0.08, 0.08, 0.15)),
)
