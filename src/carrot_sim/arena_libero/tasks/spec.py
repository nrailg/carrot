from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    asset: str
    position: tuple[float, float, float]
    scale: float | tuple[float, float, float] = 1.0
    xy_noise: float = 0.01
    body: str = ""
    joints: tuple[tuple[str, float], ...] = ()
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    asset: str
    position: tuple[float, float, float]
    joints: tuple[tuple[str, float], ...]
    scale: float | tuple[float, float, float] = 1.0
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class GoalSpec:
    kind: Literal["on_plate", "in_closed_drawer", "on_lit_stove"]
    target: str
    support: str
    support_body: str = ""
    joint: str = ""
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_size: tuple[float, float, float] = (0.08, 0.08, 0.075)
    target_half_size: tuple[float, float, float] = (0.055, 0.055, 0.025)
    closed_position: float = 0.0
    closed_tolerance: float = 0.01


@dataclass(frozen=True)
class TaskSpec:
    suite: str
    name: str
    language: str
    objects: tuple[ObjectSpec, ...]
    goal: GoalSpec
    fixtures: tuple[FixtureSpec, ...] = ()
    robot_position: tuple[float, float, float] = (2.432, -1.581, 0.75)
    camera_eye: tuple[float, float, float] = (2.48, -2.65, 1.65)
    camera_target: tuple[float, float, float] = (2.48, -2.04, 0.80)

    @property
    def task_id(self) -> str:
        return f"{self.suite}/{self.name}"

    @property
    def critic_size(self) -> int:
        return (
            26
            + 13 * (len(self.objects) + len(self.fixtures))
            + 2 * sum(len(item.joints) for item in (*self.objects, *self.fixtures))
        )

    def __post_init__(self) -> None:
        names = [obj.name for obj in self.objects] + [fixture.name for fixture in self.fixtures]
        if len(set(names)) != len(names):
            raise ValueError("Task entities must have unique names")
        if self.goal.target not in [obj.name for obj in self.objects]:
            raise ValueError("Goal target must be a movable object")
        if self.goal.support not in names:
            raise ValueError("Goal support must exist in the task scene")
        if self.goal.kind != "on_plate":
            fixture = next(item for item in self.fixtures if item.name == self.goal.support)
            if self.goal.joint not in dict(fixture.joints):
                raise ValueError("Goal joint must be declared in the support fixture")
