from carrot_sim.arena_libero.tasks.builders import (
    beside_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

desk_caddy = table_object("desk_caddy", "desk_caddy", (2.43, -2.2), scale=2.0)
book = table_object(
    "book",
    "black_book",
    (2.42, -1.97),
    scale=0.4,
    rotation=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
)
white_mug = table_object("white_mug", "white_mug", (2.18, -1.96))
red_mug = table_object("red_mug", "red_mug", (2.67, -1.96))

TASK = TaskSpec(
    suite="libero_90",
    name="L90S3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
    source_class="L90S3PickUpTheWhiteMugAndPlaceItToTheRightOfTheCaddy",
    language="Pick up the white mug and place it to the right of the caddy.",
    objects=(desk_caddy, book, white_mug, red_mug),
    goal=beside_goal(white_mug, desk_caddy, "right"),
)
