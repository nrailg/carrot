from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

desk_caddy = table_object("desk_caddy", "desk_caddy", (2.43, -2.2), scale=2.0)
book = table_object(
    "book",
    "black_book",
    (2.42, -1.97),
    scale=0.3,
    rotation=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
)
mug = table_object("mug", "yellow_white_mug", (2.67, -1.96))

TASK = TaskSpec(
    suite="libero_90",
    name="L90S1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    source_class="L90S1PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy",
    language="Pick up the book and place it in the left compartment of the caddy.",
    objects=(desk_caddy, book, mug),
    goal=object_goal(book, desk_caddy, region="left"),
)
