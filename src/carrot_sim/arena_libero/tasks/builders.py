from dataclasses import replace
from math import sqrt

from carrot_sim.arena_libero.tasks.assets import ASSETS
from carrot_sim.arena_libero.tasks.spec import FixtureSpec, GoalSpec, ObjectSpec

IDENTITY = (0.0, 0.0, 0.0, 1.0)
CABINET_SCALE = (0.604285714, 0.604016952, 0.604236970)
STOVE_SCALE = (0.992841581, 0.996457042, 1.515151515)
MOKA_SCALE = (0.560298103, 0.563758389, 0.564138413)
MICROWAVE_SCALE = (0.663931624, 0.662372517, 0.661644696)


def rotate(vector: tuple[float, float, float], quaternion: tuple[float, ...]) -> tuple[float, ...]:
    x, y, z, w = quaternion
    a, b, c = vector
    return (
        (1 - 2 * y * y - 2 * z * z) * a + 2 * (x * y - z * w) * b + 2 * (x * z + y * w) * c,
        2 * (x * y + z * w) * a + (1 - 2 * x * x - 2 * z * z) * b + 2 * (y * z - x * w) * c,
        2 * (x * z - y * w) * a + 2 * (y * z + x * w) * b + (1 - 2 * x * x - 2 * y * y) * c,
    )


def scaled_bounds(obj: ObjectSpec) -> tuple[tuple[float, ...], tuple[float, ...]]:
    geometry = ASSETS[obj.asset]
    scale = obj.scale if isinstance(obj.scale, tuple) else (obj.scale,) * 3
    return (
        tuple(a * b for a, b in zip(geometry.center, scale, strict=True)),
        tuple(a * b for a, b in zip(geometry.half_size, scale, strict=True)),
    )


def table_object(
    name: str,
    asset: str,
    xy: tuple[float, float],
    scale: float = 1.0,
    rotation: tuple[float, float, float, float] = IDENTITY,
    xy_noise: float = 0.005,
) -> ObjectSpec:
    geometry = ASSETS[asset]
    corners = (
        rotate((x * scale, y * scale, z * scale), rotation)
        for x in (geometry.minimum[0], geometry.maximum[0])
        for y in (geometry.minimum[1], geometry.maximum[1])
        for z in (geometry.minimum[2], geometry.maximum[2])
    )
    bottom = min(corner[2] for corner in corners)
    obj = ObjectSpec(
        name, asset, (*xy, 0.755 - bottom), scale=scale, rotation=rotation, xy_noise=xy_noise
    )
    if asset == "salad_dressing":
        return replace(obj, body="SaladDressing001", joints=(("SaladDressing001_Lid_joint", 0.0),))
    if asset == "ketchup_bottle":
        return replace(obj, body="Ketchup002", joints=(("Ketchup002_Lid001_joint", 0.0),))
    return obj


def cabinet(top: float = 0.0, middle: float = 0.0, bottom: float = 0.0) -> FixtureSpec:
    return FixtureSpec(
        "cabinet",
        "cabinet",
        (2.70, -2.16, 0.9131),
        tuple(
            (f"StorageFurniture136_Drawer00{index}_joint", value)
            for index, value in enumerate((top, middle, bottom), 1)
        ),
        CABINET_SCALE,
        (0.0, 0.0, -sqrt(0.5), sqrt(0.5)),
    )


def stove(on: bool = False) -> FixtureSpec:
    return FixtureSpec(
        "stove",
        "stove",
        (2.60, -2.20, 0.782),
        (("knob_center_joint", 0.8 if on else 0.0),),
        STOVE_SCALE,
        (0.0, 0.0, 1.0, 0.0),
    )


def microwave(opened: bool = False) -> FixtureSpec:
    return FixtureSpec(
        "microwave",
        "microwave",
        (2.63, -2.24, 0.8574),
        (
            ("Button001_joint", 0.0),
            ("Button002_joint", 0.0),
            ("Disc001_joint", 0.0),
            ("microjoint", 1.3 if opened else 0.0),
        ),
        MICROWAVE_SCALE,
        (0.0, 0.0, 1.0, 0.0),
    )


def moka_pot(name: str, xy: tuple[float, float]) -> ObjectSpec:
    return ObjectSpec(
        name,
        "moka_pot",
        (*xy, 0.835),
        scale=MOKA_SCALE,
        body="MokaPot001",
        joints=(("MokaPot001_Lid_joint", 0.0),),
        rotation=(0.0, 0.0, 1.0, 0.0),
    )


def winerack(name: str, xy: tuple[float, float]) -> ObjectSpec:
    return ObjectSpec(
        name,
        "wine_rack",
        (*xy, 0.9023),
        scale=(1.000023433, 0.999891415, 0.998041104),
        rotation=(0.0, 0.0, sqrt(0.5), sqrt(0.5)),
        xy_noise=0.0,
        kinematic=True,
    )


def object_goal(target: ObjectSpec, support: ObjectSpec, *, region: str = "top") -> GoalSpec:
    target_center, target_half = scaled_bounds(target)
    support_center, support_half = scaled_bounds(support)
    center = (*support_center[:2], support_center[2] + support_half[2] - 0.005)
    half = (support_half[0] * 0.75, support_half[1] * 0.75, max(0.20, target_half[2] * 3))
    kind = "on_surface"
    containment = "volume"
    if support.asset in ("plate", "small_plate"):
        kind = "on_plate"
    elif support.asset in ("bowl", "white_bowl", "ramekin") and region != "inside":
        center = (*support_center[:2], support_center[2] + 0.008)
        half = (*half[:2], support_half[2] + target_half[2] + 0.03)
    elif support.asset in ("basket", "tray", "desk_caddy") or region == "inside":
        kind, containment = "inside", "opening"
        center = support_center
        half = (support_half[0] - 0.008, support_half[1] - 0.008, support_half[2])
    if support.asset == "tray":
        scale = support.scale
        center = (0.0, 0.0, -0.011 * scale)
        half = (0.29 * scale, 0.169 * scale, 0.059 * scale)
    if support.asset == "basket":
        scale = support.scale
        center = (-0.0014 * scale, 0.0099 * scale, 0.0118 * scale)
        half = (0.095 * scale, 0.064 * scale, 0.1115 * scale)
    if support.asset == "desk_caddy":
        scale = support.scale
        regions = {
            "left": ((0.076, 0.0, 0.0), (0.035, 0.041, 0.051)),
            "right": ((-0.075, 0.0, 0.0), (0.035, 0.041, 0.051)),
            "front": ((0.0, -0.023, -0.020), (0.034, 0.019, 0.031)),
            "back": ((0.0, 0.022, 0.0), (0.034, 0.019, 0.051)),
        }
        region_center, region_half = regions[region]
        center = tuple(value * scale for value in region_center)
        half = tuple(value * scale for value in region_half)
    if support.asset == "shelf" and region in ("middle", "under"):
        scale = support.scale
        kind = "inside"
        center = (0.0, -0.003 * scale, (0.032 if region == "middle" else -0.069) * scale)
        half = (0.145 * scale, 0.080 * scale, (0.064 if region == "middle" else 0.030) * scale)
    if support.asset == "wine_rack":
        center = (0.0, -0.01, 0.035)
        half = (0.16, 0.11, 0.20)
    return GoalSpec(
        kind,
        target.name,
        support.name,
        center=center,
        half_size=half,
        target_center=target_center,
        target_half_size=target_half,
        containment=containment,
        support_half_size=support_half,
        require_contact=support.asset == "shelf" and region == "middle",
        min_upright_cos=0.95 if support.asset in ("plate", "small_plate") else -1.0,
    )


def drawer_goal(target: ObjectSpec, layer: int, *, closed: bool = False) -> GoalSpec:
    target_center, target_half = scaled_bounds(target)
    return GoalSpec(
        "in_closed_drawer" if closed else "inside",
        target.name,
        "cabinet",
        support_body=f"StorageFurniture136_Drawer00{layer}",
        joint=f"StorageFurniture136_Drawer00{layer}_joint" if closed else "",
        center=(0.000426, -0.012120, {1: 0.104, 2: 0.001, 3: -0.102442}[layer]),
        half_size=(0.150, 0.125, 0.048),
        target_center=target_center,
        target_half_size=target_half,
    )


def fixture_goal(target: ObjectSpec, fixture: str, *, lit: bool = False) -> GoalSpec:
    if target.asset == "moka_pot":
        target_center, target_half = (0.0, 0.0, 0.0), (0.042, 0.078, 0.081)
    else:
        target_center, target_half = scaled_bounds(target)
    if fixture == "stove":
        return GoalSpec(
            "on_lit_stove" if lit else "on_surface",
            target.name,
            "stove",
            support_body="Stovetop031",
            joint="knob_center_joint" if lit else "",
            center=(0.0, 0.0653, 0.02),
            half_size=(0.12, 0.12, 0.30),
            target_center=target_center,
            target_half_size=target_half,
        )
    if fixture == "microwave":
        return GoalSpec(
            "inside",
            target.name,
            "microwave",
            support_body="Microwave089",
            # 稳定杯子的完整几何底界比碰撞支撑面低约 2 mm；只给底面余量。
            # 上界仍为 0.080，XY 不变，不缩小目标物体边界。
            center=(-0.0454, 0.0154, 0.004),
            half_size=(0.12, 0.10, 0.076),
            target_center=target_center,
            target_half_size=target_half,
        )
    if fixture != "cabinet":
        raise ValueError(f"Unknown fixture: {fixture}")
    return GoalSpec(
        "on_surface",
        target.name,
        "cabinet",
        support_body="StorageFurniture136",
        center=(0.0, 0.0, 0.167),
        half_size=(0.15, 0.14, 0.30),
        target_center=target_center,
        target_half_size=target_half,
    )


def joint_goal(fixture: str, state: str, layer: int = 1) -> GoalSpec:
    if fixture == "cabinet":
        return GoalSpec(
            "joint_open" if state == "open" else "joint_closed",
            "",
            fixture,
            support_body=f"StorageFurniture136_Drawer00{layer}",
            joint=f"StorageFurniture136_Drawer00{layer}_joint",
            joint_min=0.13,
            joint_max=0.235,
        )
    if fixture == "microwave":
        return GoalSpec(
            "joint_open" if state == "open" else "joint_closed",
            "",
            fixture,
            support_body="Microwave089",
            joint="microjoint",
            joint_min=1.0,
            joint_max=1.58,
            closed_tolerance=0.05,
        )
    if fixture != "stove":
        raise ValueError(f"Unknown fixture: {fixture}")
    return GoalSpec(
        "joint_on" if state == "on" else "joint_off",
        "",
        fixture,
        support_body="Stovetop031",
        joint="knob_center_joint",
    )


def beside_goal(target: ObjectSpec, support: ObjectSpec, side: str) -> GoalSpec:
    target_center, target_half = scaled_bounds(target)
    support_center, support_half = scaled_bounds(support)
    return GoalSpec(
        "beside",
        target.name,
        support.name,
        side=side,
        target_center=target_center,
        target_half_size=target_half,
        support_half_size=support_half,
        support_center=support_center,
    )
