from carrot_sim.arena_libero.tasks.spec import FixtureSpec, GoalSpec, ObjectSpec, TaskSpec

TASK = TaskSpec(
    suite="libero_10",
    name="L10K3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    language="Turn on the stove and put the moka pot on it.",
    objects=(
        ObjectSpec(
            "moka_pot",
            "moka_pot",
            (2.27, -2.04, 0.835),
            scale=(0.56029810, 0.56375839, 0.56413841),
            rotation=(0.0, 0.0, 1.0, -0.00000367),
            body="MokaPot001",
            joints=(("MokaPot001_Lid_joint", 0.0),),
        ),
        ObjectSpec("frying_pan", "frying_pan", (2.22, -2.33, 0.81), scale=0.8),
    ),
    fixtures=(
        FixtureSpec(
            "stove",
            "stove",
            (2.60, -2.20, 0.782),
            (("knob_center_joint", 0.0),),
            scale=(0.99284158, 0.99645704, 1.51515152),
            rotation=(0.0, 0.0, 1.0, -0.00000367),
        ),
    ),
    goal=GoalSpec(
        "on_lit_stove",
        "moka_pot",
        "stove",
        support_body="Stovetop031",
        joint="knob_center_joint",
        center=(0.0, 0.0653, 0.02),
        half_size=(0.12, 0.12, 0.20),
        target_half_size=(0.042, 0.076, 0.081),
    ),
)
