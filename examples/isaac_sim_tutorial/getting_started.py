# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# 本文件复制自 Isaac Sim v5.1.0，补充中文讲解和可选录像，保留原有三轮演示逻辑。
# 教程页面：
# https://docs.isaacsim.omniverse.nvidia.com/5.1.0/introduction/quickstart_isaacsim.html
# 对应源码：
# https://github.com/isaac-sim/IsaacSim/blob/v5.1.0/source/standalone_examples/tutorials/getting_started.py
# 运行：在 Isaac Sim 安装目录执行以下命令，并替换为本仓库的实际路径。
# ./python.sh /path/to/carrot/examples/isaac_sim_tutorial/getting_started.py
# 追加 --record /path/to/getting_started.mp4 可录制视口画面，不包含窗口菜单。
#
# 页面给出 GUI、Extension、Standalone Python 三种工作流，本文件对应 Standalone Python。
# GUI 用菜单添加对象；Extension 中应用已启动，可在 Script Editor 直接运行片段。
# Standalone Python 则由脚本负责启动应用、搭建场景、推进仿真和关闭应用。
# 三轮仿真分别演示纯视觉、只有刚体属性、同时具有刚体和碰撞属性的区别。

# 保留原版 import 顺序：Isaac Sim 插件模块必须在 SimulationApp 启动后导入。
# ruff: noqa: E402, I001

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from isaacsim import SimulationApp
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description="Isaac Sim Getting Started，可选录制 MP4")
parser.add_argument("--record", type=Path, help="保存三轮演示的 MP4，30 FPS 半速播放")
args, kit_args = parser.parse_known_args()
# 自定义参数由本脚本消费，其他参数继续传给 Kit，如 --no-window 和 GPU 设置。
sys.argv = [sys.argv[0], *kit_args]
if args.record is not None:
    assert args.record.suffix.lower() == ".mp4", "--record 必须是 .mp4 路径"
    assert not args.record.exists(), f"录像文件已存在：{args.record}"

# 启动带窗口的 Isaac Sim；这里创建的是应用，不是 World 或机器人对象。
simulation_app = SimulationApp({"headless": False})

import omni.usd
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport
from pxr import Sdf, UsdLux

# Add a Ground Plane：创建 z=0 的地面，提供碰撞表面，承接后续下落的刚体。
# 对应 GUI 的 Create > Physics > Ground Plane；prim_path 是 USD 场景节点路径。
GroundPlane(prim_path="/World/GroundPlane", z_position=0)

# Add a Light Source：为物体提供照明；只有光而没有可见物体时，画面仍可能是黑的。
# 从应用的 USD Context 获取当前 stage，后面的灯光就在这个场景中创建。
stage = omni.usd.get_context().get_stage()
# 定义远光源，模拟来自固定方向的平行光；Sdf.Path 表示 USD 节点路径。
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
# 创建光照强度属性并写入 300；不会创建第二盏灯。
distantLight.CreateIntensityAttr(300)

# Add a Visual Cube：VisualCuboid 只有可视几何和外观，没有刚体、碰撞属性。
# 因此黄色方块最初不会受重力下落，也不会与其他物体发生碰撞响应。
# Core API 封装了底层 USD 操作，创建常用对象比逐项调用原生 USD 接口更简洁。
visual_cube = VisualCuboid(
    prim_path="/visual_cube",  # USD 场景中的路径，后面通过同一路径给它添加物理属性。
    name="visual_cube",  # Python 包装对象的名称，与 USD 路径是两个概念。
    position=np.array([0, 0.5, 1.0]),  # 方块中心的世界坐标，单位米。
    size=0.3,  # 未额外缩放时，方块边长为 0.3 米。
    color=np.array([255, 255, 0]),  # 保留原版的 RGB 数值，表示黄色。
)

# 创建绿色对照方块：整个示例都不向它添加刚体或碰撞属性，因此一直悬在初始位置。
# 变量名中的 static 只表示它不动，并不表示它已经具有静态碰撞体配置。
visual_cube_static = VisualCuboid(
    prim_path="/visual_cube_static",
    name="visual_cube_static",
    position=np.array([0.5, 0, 0.5]),
    size=0.3,
    color=np.array([0, 255, 0]),  # 绿色。
)

# Add Physics and Collision Properties：DynamicCuboid 创建时就附带刚体和碰撞属性。
# 刚体属性使物体参与动力学，质量、惯量等影响其运动；碰撞属性用于接触检测和响应。
# 这两类属性可以分别添加；青色方块同时具备二者，从第一轮就会下落并接触地面。
dynamic_cube = DynamicCuboid(
    prim_path="/dynamic_cube",
    name="dynamic_cube",
    position=np.array([0, -0.5, 1.5]),
    size=0.3,
    color=np.array([0, 255, 255]),  # 青色。
)

# 创建仿真管理器，指定 1 个场景长度单位为 1 米；它使用前面搭建的当前 stage。
# 上面的对象由构造函数直接加入 stage，本例没有用 my_world.scene.add() 注册它们。
my_world = World(stage_units_in_meters=1.0)

if args.record is not None:
    # 读取当前视口的渲染结果；只调整观察视角，不创建另一套物理场景。
    set_camera_view(eye=np.array([4.5, 0.0, 2.8]), target=np.array([0.0, 0.0, 0.8]))
    viewport = get_active_viewport()
    assert viewport is not None, "录像需要有效的渲染视口"
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([viewport.render_product_path])
    args.record.parent.mkdir(parents=True, exist_ok=True)
    video = imageio.get_writer(
        args.record,
        fps=30,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    phase_titles = (
        "Round 1/3 - Yellow: visual only",
        "Round 2/3 - Yellow: rigid body, no collision",
        "Round 3/3 - Yellow: rigid body + collision",
    )

# 三轮演示分别推进 100 步；对同一场景逐步添加属性，而不是创建三个独立场景。
for i in range(3):
    # 重置仿真并准备物理运行；已写入 USD 的刚体、碰撞属性不会因此被删除。
    my_world.reset()
    # i 从 0 开始，对应下面的三个阶段。
    print("simulator running", i)
    # 第二轮给已有黄色方块添加刚体属性，此时还没有给它添加碰撞属性。
    if i == 1:
        print("Adding Physics Properties to the Visual Cube")
        # 保留原版按阶段导入的结构；此处的 RigidPrim 会给节点应用刚体 API。
        from isaacsim.core.prims import RigidPrim

        # 包装已有方块并使其参与动力学，不是再创建一个方块。
        # 它开始受重力下落，但没有碰撞响应，所以会穿过地面。
        RigidPrim("/visual_cube")

    # 第三轮继续给同一黄色方块添加碰撞属性，之前的刚体属性仍然存在。
    if i == 2:
        print("Adding Collision Properties to the Visual Cube")
        from isaacsim.core.prims import GeometryPrim

        # 获取已有几何节点的包装对象，再显式应用碰撞 API。
        prim = GeometryPrim("/visual_cube")
        # 此后黄色方块既参与动力学，又能与地面发生碰撞响应。
        prim.apply_collision_apis()

    if args.record is not None:
        # 每轮只预热渲染，不推进物理，避免录像漏掉方块开始下落的过程。
        for _ in range(8):
            my_world.render()

    # 每轮推进 100 步：更新物理状态并渲染画面，才能观察到下落或保持静止的区别。
    for j in range(100):
        my_world.step(render=True)
        if args.record is not None:
            frame = rgb.get_data()
            assert frame.ndim == 3 and frame.shape[2] >= 3, f"无效录像帧：{frame.shape}"
            image = Image.fromarray(frame[:, :, :3])
            draw = ImageDraw.Draw(image)
            draw.rectangle((12, 12, 820, 58), fill=(20, 20, 20))
            draw.text((24, 20), f"{phase_titles[i]} | step {j + 1}/100", fill="white", font_size=24)
            video.append_data(np.asarray(image))

if args.record is not None:
    # 默认物理步长为 1/60 秒，每步录一帧、以 30 FPS 播放，方便观察三轮差异。
    video.close()
    rgb.detach()
    print(f"Recorded 300 frames: {args.record}", flush=True)

# 页面还介绍 Move, Rotate, and Scale：GUI 用 W/E/R 切换平移、旋转、缩放工具，
# 可在 Property 面板输入精确值；平移、旋转需区分世界坐标和局部坐标。
# 原版 getting_started.py 没有执行这一部分，以上保留其三轮物理演示。
# 页面末尾的 Standalone 变换片段与 5.1 API 存在不一致，阅读时以接口文档为准：
# XFormPrim.set_world_poses 的位置形状为 (N, 3)，朝向为 (N, 4) 的 (w, x, y, z) 四元数，
# 不能把三个欧拉角或“弧度数组”直接当作它的朝向参数；缩放方法为 set_local_scales。
# 原生 USD 的 AddRotateXYZOp 使用角度制欧拉角，不能与上述四元数接口混淆。
# https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.prims/docs/index.html

# 三轮共 300 步完成后自动关闭应用；原版不会一直等待用户手动关闭窗口。
simulation_app.close()
