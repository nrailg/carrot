from dataclasses import replace
from math import sqrt

from carrot_sim.arena_libero.tasks.builders import (
    fixture_goal,
    joint_goal,
    microwave,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

porcelain_mug = table_object("porcelain_mug", "white_mug", (2.18, -2.1))
white_yellow_mug = table_object(
    "white_yellow_mug",
    "yellow_white_mug",
    (2.42, -1.91),
    rotation=(0.0, 0.0, -sqrt(0.5), sqrt(0.5)),
)

TASK = TaskSpec(
    suite="libero_10",
    name="L10K6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    source_class="L10K6PutTheYellowAndWhiteMugInTheMicrowaveAndCloseIt",
    language="Put the yellow and white mug in the microwave and close it.",
    objects=(porcelain_mug, white_yellow_mug),
    fixtures=(microwave(opened=True),),
    goal=fixture_goal(white_yellow_mug, "microwave"),
    goals=(replace(joint_goal("microwave", "closed"), release_distance=0.4),),
)
