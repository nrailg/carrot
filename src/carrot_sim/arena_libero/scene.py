from typing import override

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import DomeLightCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_arena.scene.scene import Scene

from carrot_sim.arena_libero.config import ArenaLiberoConfig


@configclass
class KitchenCfg:
    kitchen: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Kitchen")
    bowl: RigidObjectCfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Bowl")
    plate: RigidObjectCfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Plate")
    light = AssetBaseCfg(prim_path="/World/Light", spawn=DomeLightCfg(intensity=1500.0))
    bowl_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Bowl/Bowl008",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Plate/Plate012"],
        update_period=0.0,
    )


class KitchenScene(Scene):
    def __init__(self, config: ArenaLiberoConfig) -> None:
        super().__init__()
        self.config = KitchenCfg()
        self.config.kitchen.spawn = UsdFileCfg(usd_path=config.asset("scene"))
        for name, pos in (("bowl", (2.36, -2.00, 0.815)), ("plate", (2.58, -2.14, 0.79))):
            item = self.config.bowl if name == "bowl" else self.config.plate
            item.spawn = UsdFileCfg(usd_path=config.asset(name), activate_contact_sensors=True)
            item.init_state.pos = pos

    @override
    def get_scene_cfg(self) -> KitchenCfg:
        return self.config
