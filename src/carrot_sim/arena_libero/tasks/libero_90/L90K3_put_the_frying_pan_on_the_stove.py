from carrot_sim.arena_libero.tasks.builders import (
    fixture_goal,
    stove,
    table_object,
)
from carrot_sim.arena_libero.tasks.builders import (
    moka_pot as moka_pot_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

chefmate_8_frypan = table_object(
    "chefmate_8_frypan",
    "frying_pan",
    (2.13, -2.04),
    rotation=(0.0, 0.0, -0.7071067811865476, 0.7071067811865476),
)
moka_pot = moka_pot_object("moka_pot", (2.42, -1.92))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K3_put_the_frying_pan_on_the_stove",
    source_class="L90K3PutTheFryingPanOnTheStove",
    language="Put the frying pan on the stove.",
    objects=(chefmate_8_frypan, moka_pot),
    fixtures=(stove(),),
    goal=fixture_goal(chefmate_8_frypan, "stove"),
)
