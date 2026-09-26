from typing import override

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import (
    ArticulationRootPropertiesCfg,
    DomeLightCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
)
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.configclass import make_configclass
from pxr import Usd, UsdPhysics

from carrot_sim.arena_libero.config import ArenaLiberoConfig
from carrot_sim.arena_libero.tasks.spec import TaskSpec


def rigid_body_path(asset_path: str, body_name: str = "") -> str:
    stage = Usd.Stage.Open(asset_path)
    root = stage.GetDefaultPrim()
    bodies = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if body_name:
        bodies = [body for body in bodies if body.GetName() == body_name]
    if len(bodies) != 1:
        raise ValueError(f"Expected one rigid body in {asset_path}, got {len(bodies)}")
    path = bodies[0].GetPath().MakeRelativePath(root.GetPath())
    return "" if str(path) == "." else f"/{path}"


class TaskScene(Scene):
    def __init__(self, config: ArenaLiberoConfig, spec: TaskSpec) -> None:
        super().__init__()
        fields = [
            (
                "kitchen",
                AssetBaseCfg,
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/Kitchen",
                    spawn=UsdFileCfg(usd_path=config.asset("scene")),
                ),
            ),
            (
                "light",
                AssetBaseCfg,
                AssetBaseCfg(
                    prim_path="/World/Light",
                    spawn=DomeLightCfg(intensity=1500.0),
                ),
            ),
        ]
        paths = {}
        for obj in spec.objects:
            path = config.asset(obj.asset)
            paths[obj.name] = "{ENV_REGEX_NS}/" + obj.name + rigid_body_path(path, obj.body)
            spawn = UsdFileCfg(
                usd_path=path,
                scale=obj.scale if isinstance(obj.scale, tuple) else (obj.scale,) * 3,
                activate_contact_sensors=True,
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True)
                if obj.kinematic
                else None,
            )
            if obj.joints:
                spawn.articulation_props = ArticulationRootPropertiesCfg(
                    fix_root_link=False, articulation_enabled=True
                )
                item = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/" + obj.name,
                    spawn=spawn,
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=obj.position,
                        rot=obj.rotation,
                        joint_pos=dict(obj.joints),
                    ),
                    actuators={
                        "passive": ImplicitActuatorCfg(
                            joint_names_expr=[name for name, _ in obj.joints],
                            stiffness=0.0,
                            damping=0.1,
                        )
                    },
                )
            else:
                item = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/" + obj.name,
                    spawn=spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=obj.position, rot=obj.rotation),
                )
            fields.append((obj.name, type(item), item))
        for fixture in spec.fixtures:
            fields.append(
                (
                    fixture.name,
                    ArticulationCfg,
                    ArticulationCfg(
                        prim_path="{ENV_REGEX_NS}/" + fixture.name,
                        spawn=UsdFileCfg(
                            usd_path=config.asset(fixture.asset),
                            scale=fixture.scale
                            if isinstance(fixture.scale, tuple)
                            else (fixture.scale,) * 3,
                            activate_contact_sensors=True,
                            articulation_props=ArticulationRootPropertiesCfg(
                                fix_root_link=True, articulation_enabled=True
                            ),
                        ),
                        init_state=ArticulationCfg.InitialStateCfg(
                            pos=fixture.position,
                            rot=fixture.rotation,
                            joint_pos=dict(fixture.joints),
                        ),
                        actuators={
                            "passive": ImplicitActuatorCfg(
                                joint_names_expr=[name for name, _ in fixture.joints],
                                stiffness=0.0,
                                damping=0.1,
                            )
                        },
                    ),
                )
            )
        for index, goal in enumerate(spec.conditions):
            if (
                goal.kind not in ("on_plate", "on_lit_stove", "on_surface")
                and goal.containment != "opening"
                and not goal.require_contact
            ):
                continue
            if goal.support_body:
                support = next(
                    item for item in (*spec.objects, *spec.fixtures) if item.name == goal.support
                )
                support_path = (
                    "{ENV_REGEX_NS}/"
                    + goal.support
                    + rigid_body_path(config.asset(support.asset), goal.support_body)
                )
            else:
                support_path = paths[goal.support]
            fields.append(
                (
                    f"target_contact_{index}",
                    ContactSensorCfg,
                    ContactSensorCfg(
                        prim_path=paths[goal.target],
                        filter_prim_paths_expr=[support_path],
                        update_period=0.0,
                    ),
                )
            )
        self.config = make_configclass("LiberoSceneCfg", fields)()

    @override
    def get_scene_cfg(self) -> object:
        return self.config
