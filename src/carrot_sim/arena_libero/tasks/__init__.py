from carrot_sim.arena_libero.tasks.libero_10 import TASKS as LIBERO_10
from carrot_sim.arena_libero.tasks.libero_90 import TASKS as LIBERO_90
from carrot_sim.arena_libero.tasks.libero_goal import TASKS as LIBERO_GOAL
from carrot_sim.arena_libero.tasks.libero_object import TASKS as LIBERO_OBJECT
from carrot_sim.arena_libero.tasks.libero_spatial import TASKS as LIBERO_SPATIAL
from carrot_sim.arena_libero.tasks.spec import TaskSpec

_ALL_TASKS = (*LIBERO_SPATIAL, *LIBERO_OBJECT, *LIBERO_GOAL, *LIBERO_10, *LIBERO_90)
_TASKS = {task.task_id: task for task in _ALL_TASKS}
if len(_TASKS) != len(_ALL_TASKS):
    raise ValueError("Duplicate LIBERO task IDs")


def list_tasks(suite: str | None = None) -> tuple[TaskSpec, ...]:
    """List immutable task definitions without importing Isaac or starting Kit.

    Parameters
    ----------
    suite : str, optional
        Restrict to one of the five LIBERO suites; unknown suites raise KeyError.

    Returns
    -------
    tuple[TaskSpec, ...]
        Definitions in registry order, with stable suite/name task IDs.
    """
    if suite is not None and suite not in {task.suite for task in _TASKS.values()}:
        raise KeyError(f"Unknown suite: {suite}")
    return tuple(task for task in _TASKS.values() if suite is None or task.suite == suite)


def get_task(task_id: str) -> TaskSpec:
    """Resolve an exact suite/name task ID before allocating simulation resources.

    Parameters
    ----------
    task_id : str
        One of the IDs returned by list_tasks(); no fallback for unknown IDs.

    Returns
    -------
    TaskSpec
        Immutable task definition.
    """
    return _TASKS[task_id]
