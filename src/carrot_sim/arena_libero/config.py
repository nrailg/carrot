from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArenaLiberoConfig:
    """Configure the bowl-on-plate environment using explicit local USD assets.

    Parameters
    ----------
    asset_root : Path
        Directory containing scene.usd, bowl.usd and plate.usd, including
        their referenced files. These may be symlinks to an existing asset cache.
    num_envs : int
        Parallel environments in one simulator process, with one renderer per GPU.
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

    def __post_init__(self) -> None:
        for name in ("num_envs", "image_size", "max_episode_steps"):
            value = self.__dict__[name]
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def asset(self, name: str) -> str:
        path = (self.asset_root / f"{name}.usd").resolve(strict=True)
        return str(path)
