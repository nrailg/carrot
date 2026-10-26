# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demonstrate robot articulation control with Franka arm and Nova Carter."""

# 源码：Isaac Sim v6.0.1，保留原版四轮关节控制，可选录制当前视口。
# https://github.com/isaac-sim/IsaacSim/blob/v6.0.1/source/standalone_examples/tutorials/getting_started/getting_started_robot.py
# 运行：bash examples/isaac_sim_tutorial/test_getting_started_robot.sh --record /path/to/robot.mp4
# Isaac Sim 插件必须在 SimulationApp 启动后导入。
# ruff: noqa: E402, I001

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from isaacsim import SimulationApp
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description="Isaac Sim Robot，可选录制 MP4")
parser.add_argument("--record", type=Path, help="保存四轮演示的 MP4，30 FPS 半速播放")
args, kit_args = parser.parse_known_args()
# 自定义参数由脚本消费，其余 GPU、窗口等参数继续交给 Kit。
sys.argv = [sys.argv[0], *kit_args]
if args.record is not None:
    assert args.record.suffix.lower() == ".mp4", "--record 必须是 .mp4 路径"
    assert not args.record.exists(), f"录像文件已存在：{args.record}"

simulation_app = SimulationApp({"headless": False})  # - Start the simulation app, with GUI open

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.replicator.core as rep
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from omni.kit.viewport.utility import get_active_viewport

# Preparing the scene.
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

# Create a new stage
stage_utils.create_new_stage()
stage_utils.set_stage_units(meters_per_unit=1.0)

# Add ground plane
GroundPlane("/World/GroundPlane", positions=[0, 0, 0])

# Add distant light
distant_light = DistantLight("/World/DistantLight")
distant_light.set_intensities(300)

# Set camera view
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)

# Add Franka
asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
stage_utils.add_reference_to_stage(
    usd_path=asset_path,
    path="/World/Arm",
)  # - Add robot to stage

# Set initial pose before creating articulation.
arm_transform = XformPrim("/World/Arm")
arm_transform.set_world_poses(positions=[0.0, 1.0, 0.0])

arm = Articulation("/World/Arm")  # - Create an articulation object `arm`

# Add Carter
asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
stage_utils.add_reference_to_stage(
    usd_path=asset_path,
    path="/World/Car",
)

# Set initial pose before creating articulation.
car_transform = XformPrim("/World/Car")
car_transform.set_world_poses(positions=[0.0, -1.0, 0.0])

car = Articulation("/World/Car")  # - Create an articulation object `car`

# Initialize physics
SimulationManager.set_physics_dt(1.0 / 60.0)

if args.record is not None:
    set_camera_view(eye=[5.0, 0.0, 3.0], target=[0.0, 0.0, 0.7])
    # 新 stage 的视口在应用更新后才重建；先更新，避免绑定已失效的渲染产品。
    for _ in range(8):
        RenderingManager.render()
    viewport = get_active_viewport()
    assert viewport is not None, "录像需要有效的渲染视口"
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([viewport.render_product_path])
    rep.orchestrator.set_capture_on_play(False)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    video = imageio.get_writer(
        args.record,
        fps=30,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    phase_titles = (
        "Round 1/4 - Initial state",
        "Round 2/4 - Move arm and Carter",
        "Round 3/4 - Reset arm and stop Carter",
        "Round 4/4 - Move and read joint positions",
    )
    # 时间线尚未启动，仅预热渲染，避免漏掉初始状态。
    for _ in range(8):
        RenderingManager.render()
        simulation_app.update()

# Start timeline
app_utils.play()
simulation_app.update()

for i in range(4):
    print("Running cycle:", i)
    if i == 1 or i == 3:
        print("Moving")
        # - Move the `arm`
        arm.set_dof_positions([-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04])
        # - Move the `car`
        car.set_dof_velocities([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    if i == 2:
        print("Stopping")
        # - Reset the `arm`
        arm.set_dof_positions([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # - Stop the `car`
        car.set_dof_velocities([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for j in range(100):
        # - Step the simulation, both rendering and physics.
        SimulationManager.step()
        RenderingManager.render()
        simulation_app.update()
        if args.record is not None:
            # 显式触发采集；delta_time=0 不额外推进物理，也不暂停四轮演示的时间线。
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
            frame = rgb.get_data()
            assert frame.ndim == 3 and frame.shape[2] >= 3, f"无效录像帧：{frame.shape}"
            image = Image.fromarray(frame[:, :, :3])
            draw = ImageDraw.Draw(image)
            draw.rectangle((12, 12, 960, 58), fill=(20, 20, 20))
            draw.text((24, 20), f"{phase_titles[i]} | step {j + 1}/100", fill="white", font_size=24)
            video.append_data(np.asarray(image))
        # - Print the joint positions of `car` at every physics step
        if i == 3:
            car_joint_positions = car.get_dof_positions()
            print("Car joint positions:", car_joint_positions)

# Stop timeline before closing
app_utils.stop()
if args.record is not None:
    # 400 个物理步以 30 FPS 播放，便于观察四轮控制差异。
    video.close()
    rgb.detach()
    print(f"Recorded 400 frames: {args.record}", flush=True)
simulation_app.close()
