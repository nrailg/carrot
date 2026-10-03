from dataclasses import dataclass
from pathlib import Path

from carrot_sim.arena_libero.tasks import get_task


@dataclass(frozen=True)
class ArenaLiberoConfig:
    """Configure a LIBERO task using explicit local USD assets.

    Parameters
    ----------
    asset_root : Path
        Directory containing scene.usd and the selected task's named USD assets,
        including referenced files. Symlinks to an existing cache are supported.
    task_id : str, optional
        Exact suite/name from tasks.list_tasks(). None selects the original
        two-object bowl-on-plate smoke environment.
    num_envs : int
        Parallel environments in one simulator process, with one renderer per GPU.
    physics_substeps : int
        Physics steps per 20 ms control step; increasing this preserves the RL clock.
    max_episode_steps : int
        Time limit in control steps; truncations bootstrap from the final observation.
    """

    asset_root: Path
    panda_usd: str = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
        "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
    )
    num_envs: int = 4
    device: str = "cuda:0"
    seed: int = 42
    image_size: int = 256
    max_episode_steps: int = 400
    task_id: str | None = None
    physics_substeps: int = 2

    def __post_init__(self) -> None:
        if self.task_id is not None:
            get_task(self.task_id)
        for name in ("num_envs", "image_size", "max_episode_steps", "physics_substeps"):
            value = self.__dict__[name]
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def asset(self, name: str) -> str:
        path = (self.asset_root / f"{name}.usd").resolve(strict=True)
        return str(path)
