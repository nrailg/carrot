from carrot_sim.arena_libero.tasks.spec import FixtureSpec, GoalSpec, ObjectSpec, TaskSpec

TASK = TaskSpec(
    source_class="LSPickUpBlackBowlInTopDrawerOfWoodenCabinetAndPlaceItOnPlate",
    suite="libero_spatial",
    name="LS_pick_up_black_bowl_in_top_drawer_of_wooden_cabinet_and_place_it_on_plate",
    language=(
        "Pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate."
    ),
    objects=(
        ObjectSpec("bowl_target", "bowl", (2.48, -2.16, 1.0), scale=0.6, xy_noise=0.005),
        ObjectSpec("plate", "plate", (2.25, -2.14, 0.775), scale=0.8),
        ObjectSpec("ramekin", "ramekin", (2.21, -1.89, 0.80), scale=0.5),
        ObjectSpec("bowl_distractor", "bowl", (2.66, -1.8, 0.787), scale=0.6),
        ObjectSpec("cookies", "cookies", (2.20, -2.36, 0.80)),
    ),
    fixtures=(
        FixtureSpec(
            "cabinet",
            "cabinet",
            (2.70, -2.16, 0.9131),
            (
                ("StorageFurniture136_Drawer001_joint", 0.18),
                ("StorageFurniture136_Drawer002_joint", 0.0),
                ("StorageFurniture136_Drawer003_joint", 0.0),
            ),
            scale=(0.60428571, 0.60401695, 0.60423697),
            rotation=(0.0, 0.0, -0.70710808, 0.70710548),
        ),
    ),
    goal=GoalSpec("on_plate", "bowl_target", "plate", half_size=(0.08, 0.08, 0.15)),
)
