from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

shelf = table_object("shelf", "shelf", (2.41, -2.2), rotation=(0.0, 0.0, 1.0, 0.0))
black_book = table_object(
    "black_book",
    "black_book",
    (2.65, -1.98),
    scale=0.4,
    rotation=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
)
yellow_book = table_object(
    "yellow_book",
    "yellow_book",
    (2.16, -1.98),
    scale=0.4,
    rotation=(0.0, 0.7071067811865476, 0.0, 0.7071067811865476),
)
midlle_book = table_object(
    "midlle_book",
    "yellow_book",
    (2.41, -1.98),
    scale=0.4,
    rotation=(0.0, 0.7071067811865476, 0.0, 0.7071067811865476),
)

TASK = TaskSpec(
    suite="libero_90",
    name="L90S4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
    source_class="L90S4PickUpTheBookOnTheRightAndPlaceItOnTheCabinetShelf",
    language="Pick up the book on the right and place it on the cabinet shelf.",
    objects=(shelf, black_book, yellow_book, midlle_book),
    goal=object_goal(yellow_book, shelf, region="middle"),
)
