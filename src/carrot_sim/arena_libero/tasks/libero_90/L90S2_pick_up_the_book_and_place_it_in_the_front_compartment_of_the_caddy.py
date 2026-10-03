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
    scale=0.4,
    rotation=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
)
mug = table_object("mug", "red_mug", (2.67, -1.96))

TASK = TaskSpec(
    suite="libero_90",
    name="L90S2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    source_class="L90S2PickUpTheBookAndPlaceItInTheFrontCompartmentOfTheCaddy",
    language="Pick up the book and place it in the front compartment of the caddy.",
    objects=(desk_caddy, book, mug),
    goal=object_goal(book, desk_caddy, region="front"),
)
