# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
from contextlib import nullcontext
from pathlib import Path

from asset_root import configure_asset_root_from_env

configure_asset_root_from_env()

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument("--num_steps", type=int, default=0, help="仿真步数；0 表示持续运行")
parser.add_argument("--record", type=Path, help="保存 25 FPS、640×480 的 MP4 和首帧 PNG")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
assert args_cli.num_steps >= 0, "num_steps 必须非负"
if args_cli.record is not None:
    assert args_cli.record.suffix.lower() == ".mp4", "record 必须是 .mp4 路径"
    assert args_cli.num_steps >= 4, "录像时 num_steps 至少为 4"
    args_cli.enable_cameras = True
# 遥测发送进程在当前容器无法启动，排除该扩展不影响物理仿真。
args_cli.kit_args += " --/app/extensions/excluded/0=omni.kit.telemetry"
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

configure_asset_root_from_env()

import imageio.v2 as imageio
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"
    )
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    sim_utils.create_prim("/World/Objects", "Xform")
    # spawn a red cone
    cfg_cone = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

    # spawn a green cone with colliders and rigid body
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )

    # spawn a blue cuboid with deformable body
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))

    # spawn a usd file of a table into the scene
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    # Design scene
    design_scene()
    camera = None
    if args_cli.record is not None:
        camera = Camera(
            CameraCfg(
                prim_path="/World/RecordingCamera",
                height=480,
                width=640,
                data_types=["rgb"],
                update_period=0.0,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0, clipping_range=(0.05, 100.0)
                ),
            )
        )
    # Play the simulator
    sim.reset()
    if camera is not None:
        camera.set_world_poses_from_view(eyes=[[3.0, 3.0, 3.0]], targets=[[-0.3, 0.0, 0.9]])
        # 只预热渲染，不推进物理，避免录像漏掉自由落体的起始过程。
        for _ in range(8):
            sim.render()
        args_cli.record.parent.mkdir(parents=True, exist_ok=True)
    # Now we are ready!
    print("[INFO]: Setup complete...", flush=True)

    # dt=0.01，每四个物理步保存一帧，视频时间与仿真时间一致（25 FPS）。
    video = (
        imageio.get_writer(args_cli.record, fps=25, codec="libx264", pixelformat="yuv420p")
        if camera is not None
        else nullcontext()
    )
    step = 0
    frames = 0
    with video as writer:
        while simulation_app.is_running() and (args_cli.num_steps == 0 or step < args_cli.num_steps):
            sim.step()
            step += 1
            if camera is not None and step % 4 == 0:
                camera.update(4 * sim_cfg.dt, force_recompute=True)
                rgb = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
                writer.append_data(rgb)
                if frames == 0:
                    imageio.imwrite(args_cli.record.with_suffix(".png"), rgb)
                frames += 1
            if step % 100 == 0:
                print(f"[INFO]: Simulation step={step}, time={step * sim_cfg.dt:.2f}s", flush=True)
    print(f"[INFO]: Simulation finished: {step} steps", flush=True)
    if camera is not None:
        print(f"[INFO]: Recorded {frames} frames: {args_cli.record}", flush=True)


if __name__ == "__main__":
    # run the main function
    try:
        main()
    finally:
        simulation_app.close()