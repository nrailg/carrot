from carrot_sim.arena_libero.tasks.libero_10 import TASKS as LIBERO_10
from carrot_sim.arena_libero.tasks.libero_spatial import TASKS as LIBERO_SPATIAL
from carrot_sim.arena_libero.tasks.spec import TaskSpec

_TASKS = {task.task_id: task for task in (*LIBERO_SPATIAL, *LIBERO_10)}


def list_tasks(suite: str | None = None) -> tuple[TaskSpec, ...]:
    """List immutable task definitions without importing Isaac or starting Kit.

    Parameters
    ----------
    suite : str, optional
        Restrict to libero_spatial or libero_10; unknown suites raise KeyError.

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
