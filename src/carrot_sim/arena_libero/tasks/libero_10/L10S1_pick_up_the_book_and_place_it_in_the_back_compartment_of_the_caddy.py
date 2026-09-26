from dataclasses import replace
from math import sqrt

from carrot_sim.arena_libero.tasks.builders import object_goal, table_object
from carrot_sim.arena_libero.tasks.spec import TaskSpec

black_book = table_object(
    "black_book",
    "black_book",
    (2.42, -1.97),
    scale=0.4,
    rotation=(sqrt(0.5), 0.0, 0.0, sqrt(0.5)),
)
desk_caddy = table_object("desk_caddy", "desk_caddy", (2.43, -2.2), scale=2.0)
white_yellow_mug = table_object("white_yellow_mug", "yellow_white_mug", (2.67, -1.96))

TASK = TaskSpec(
    suite="libero_10",
    name="L10S1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    source_class="L10S1PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy",
    language="Pick up the book and place it in the back compartment of the caddy.",
    objects=(black_book, desk_caddy, white_yellow_mug),
    goal=replace(object_goal(black_book, desk_caddy, region="back"), release_distance=0.35),
)
