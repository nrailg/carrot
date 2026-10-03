from dataclasses import replace

from carrot_sim.arena_libero.tasks.builders import fixture_goal, moka_pot, stove
from carrot_sim.arena_libero.tasks.spec import TaskSpec

moka_pot_left = moka_pot("moka_pot_left", (2.38, -1.94))
moka_pot_right = moka_pot("moka_pot_right", (2.17, -2.06))

TASK = TaskSpec(
    suite="libero_10",
    name="L10K8_put_both_moka_pots_on_the_stove",
    source_class="L10K8PutBothMokaPotsOnTheStove",
    language="Put both moka pots on the stove.",
    objects=(moka_pot_left, moka_pot_right),
    fixtures=(stove(),),
    goal=replace(
        fixture_goal(moka_pot_left, "stove"),
        target_half_size=(0.042, 0.078, 0.081),
        release_distance=0.25,
        min_upright_cos=0.8,
    ),
    goals=(
        replace(
            fixture_goal(moka_pot_right, "stove"),
            target_half_size=(0.042, 0.078, 0.081),
            release_distance=0.25,
            min_upright_cos=0.8,
        ),
    ),
)
