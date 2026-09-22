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
    kinematic: bool = False


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
    kind: Literal[
        "on_plate",
        "in_closed_drawer",
        "on_lit_stove",
        "inside",
        "on_surface",
        "relative",
        "beside",
        "joint_open",
        "joint_closed",
        "joint_on",
        "joint_off",
    ]
    target: str
    support: str
    support_body: str = ""
    joint: str = ""
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_size: tuple[float, float, float] = (0.08, 0.08, 0.075)
    target_half_size: tuple[float, float, float] = (0.055, 0.055, 0.025)
    closed_position: float = 0.0
    closed_tolerance: float = 0.01
    joint_min: float = 0.35
    joint_max: float = 5.933185307179587
    frame: Literal["support", "world"] = "support"
    containment: Literal["volume", "opening", "partial"] = "volume"
    overlap_fraction: float = 0.1
    require_contact: bool = False
    target_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    release_distance: float = 0.1
    min_upright_cos: float = -1.0
    side: Literal["left", "right", "front", "back"] = "right"
    edge_distance: tuple[float, float] = (0.001, 0.1)
    side_tolerance: float = 0.25
    support_half_size: tuple[float, float, float] = (0.1, 0.1, 0.01)
    support_center: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TaskSpec:
    suite: str
    name: str
    language: str
    objects: tuple[ObjectSpec, ...]
    goal: GoalSpec
    fixtures: tuple[FixtureSpec, ...] = ()
    goals: tuple[GoalSpec, ...] = ()
    source_class: str = ""
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
        for goal in self.conditions:
            if not goal.kind.startswith("joint_") and goal.target not in [
                obj.name for obj in self.objects
            ]:
                raise ValueError("Goal target must be a movable object")
            if goal.support not in names:
                raise ValueError("Goal support must exist in the task scene")
            if goal.joint:
                fixture = next(
                    item for item in (*self.objects, *self.fixtures) if item.name == goal.support
                )
                if goal.joint not in dict(fixture.joints):
                    raise ValueError("Goal joint must be declared in the support fixture")

    @property
    def conditions(self) -> tuple[GoalSpec, ...]:
        return (self.goal, *self.goals)
