from carrot_sim.arena_libero.tasks.builders import (
    fixture_goal,
    stove,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

shelf = table_object("shelf", "shelf", (2.2, -2.27), scale=0.8, rotation=(0.0, 0.0, 1.0, 0.0))
bowl = table_object("bowl", "ramekin", (2.81, -1.91))
frying_pan = table_object(
    "frying_pan",
    "frying_pan",
    (2.12, -1.96),
    scale=0.8,
    rotation=(0.0, 0.0, -0.7071067811865476, 0.7071067811865476),
)

TASK = TaskSpec(
    suite="libero_90",
    name="L90K9_turn_on_the_stove_and_put_the_frying_pan_on_it",
    source_class="L90K9TurnOnTheStoveAndPutTheFryingPanOnIt",
    language="Turn on the stove and put the frying pan on it.",
    objects=(shelf, bowl, frying_pan),
    fixtures=(stove(),),
    goal=fixture_goal(frying_pan, "stove", lit=True),
)
