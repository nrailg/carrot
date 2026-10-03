from carrot_sim.arena_libero.tasks.spec import FixtureSpec, GoalSpec, ObjectSpec, TaskSpec

TASK = TaskSpec(
    source_class="L10K4PutTheBlackBowlInTheBottomDrawerOfTheCabinetAndCloseIt",
    suite="libero_10",
    name="L10K4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    language="Put the black bowl in the bottom drawer of the cabinet and close it.",
    objects=(
        ObjectSpec("bowl_target", "bowl", (2.30, -2.0, 0.791), scale=0.7),
        ObjectSpec("wine_bottle", "bottle", (2.27, -2.28, 0.90), scale=0.8),
    ),
    fixtures=(
        FixtureSpec(
            "cabinet",
            "cabinet",
            (2.70, -2.16, 0.9131),
            (
                ("StorageFurniture136_Drawer001_joint", 0.0),
                ("StorageFurniture136_Drawer002_joint", 0.0),
                ("StorageFurniture136_Drawer003_joint", 0.20),
            ),
            scale=(0.60428571, 0.60401695, 0.60423697),
            rotation=(0.0, 0.0, -0.70710808, 0.70710548),
        ),
    ),
    goal=GoalSpec(
        "in_closed_drawer",
        "bowl_target",
        "cabinet",
        support_body="StorageFurniture136_Drawer003",
        joint="StorageFurniture136_Drawer003_joint",
        center=(0.000426, -0.012120, -0.102442),
        half_size=(0.150, 0.125, 0.048),
        target_half_size=(0.064, 0.064, 0.029),
    ),
)
