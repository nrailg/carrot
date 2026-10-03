from carrot_sim.arena_libero.tasks.builders import (
    cabinet,
    object_goal,
    table_object,
    winerack,
)
from carrot_sim.arena_libero.tasks.spec import TaskSpec

akita_black_bowl = table_object("akita_black_bowl", "bowl", (2.41, -1.9))
wine_bottle = table_object("wine_bottle", "bottle", (2.16, -1.90), scale=0.8)
wine_rack = winerack("wine_rack", (2.15, -2.20))

TASK = TaskSpec(
    suite="libero_90",
    name="L90K4_put_the_wine_bottle_on_the_wine_rack",
    source_class="L90K4PutTheWineBottleOnTheWineRack",
    language="Put the wine bottle on the wine rack.",
    objects=(akita_black_bowl, wine_bottle, wine_rack),
    fixtures=(cabinet(),),
    goal=object_goal(wine_bottle, wine_rack),
)
