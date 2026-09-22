from carrot_sim.arena_libero.tasks.builders import (
    object_goal,
    table_object,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

wooden_two_layer_shelf = table_object(
    "wooden_two_layer_shelf", "shelf", (2.41, -2.2), scale=1.2, rotation=(0.0, 0.0, 1.0, 0.0)
)
black_book = table_object(
    "black_book",
    "black_book",
    (2.65, -1.98),
    scale=0.4,
    rotation=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
)
yellow_book1 = table_object(
    "yellow_book1",
    "yellow_book",
    (2.41, -1.98),
    scale=0.4,
    rotation=(0.0, 0.7071067811865476, 0.0, 0.7071067811865476),
)
yellow_book = table_object(
    "yellow_book",
    "yellow_book",
    (2.16, -1.98),
    scale=0.4,
    rotation=(0.0, 0.7071067811865476, 0.0, 0.7071067811865476),
)

TASK = TaskSpec(
    suite="libero_90",
    name="L90S4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
    source_class="L90S4PickUpTheBookInTheMiddleAndPlaceItOnTheCabinetShelf",
    language="Pick up the book in the middle and place it on the cabinet shelf.",
    objects=(wooden_two_layer_shelf, black_book, yellow_book1, yellow_book),
    goal=object_goal(yellow_book1, wooden_two_layer_shelf, region="middle"),
)
