from carrot_sim.arena_libero.tasks.builders import (
    joint_goal,
    stove,
)
from carrot_sim.arena_libero.tasks.builders import (
    moka_pot as moka_pot_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

moka_pot_left = moka_pot_object("moka_pot_left", (2.38, -1.94))
moka_pot_right = moka_pot_object("moka_pot_right", (2.17, -2.06))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K8_turn_off_the_stove",
    source_class="L90K8TurnOffTheStove",
    language="Turn off the stove.",
    objects=(moka_pot_left, moka_pot_right),
    fixtures=(stove(on=True),),
    goal=joint_goal("stove", "off"),
)
