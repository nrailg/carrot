"""使用 Isaac Sim 创建一个会自由落体的刚体方块。"""

# ruff: noqa: E402, I001

from isaacsim import SimulationApp


# Isaac Sim 的其他模块由插件系统提供，必须先启动应用才能导入。
simulation_app = SimulationApp({"headless": False})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim


# Stage 可以理解为整个三维世界，所有物体都放在 /World 下面。
stage_utils.create_new_stage()
GroundPlane("/World/Ground", positions=[0.0, 0.0, 0.0])

# 没有光源时窗口里的物体会很暗；灯光只影响显示，不影响物理。
light = DistantLight("/World/Light")
light.set_intensities(500)

# Cube 先创建几何外形：边长 0.2 米，中心位于地面上方 1 米。
cube_shape = Cube(
    paths="/World/Cube",
    positions=[0.0, 0.0, 1.0],
    sizes=0.2,
)

# RigidPrim 让方块受重力影响；GeomPrim 让它能和地面发生碰撞。
cube = RigidPrim(paths=cube_shape.paths, masses=[1.0])
cube_collision = GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)

# 开始播放时间线。第一次 update 会初始化物理场景和刚体状态。
app_utils.play()
simulation_app.update()

for step in range(300):
    simulation_app.update()

    # 每 30 帧打印一次高度，方便从数值上观察自由落体和落地过程。
    if step % 30 == 0:
        positions, _ = cube.get_world_poses()
        print(f"step={step:3d}, cube position={positions[0]}")

# 显式关闭应用，否则 Python 进程可能等待 Isaac Sim 后台线程退出。
simulation_app.close()
