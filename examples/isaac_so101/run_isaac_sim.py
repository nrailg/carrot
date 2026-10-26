"""Load one SO101 and continuously sweep its shoulder-pan joint."""

# ruff: noqa: E402, I001

import argparse
import os
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--robot-usd", type=Path, required=True)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--amplitude", type=float, default=30.0, help="Sweep amplitude in degrees")
parser.add_argument("--period", type=float, default=4.0, help="Full cycle in simulation seconds")
parser.add_argument("--steps", type=int, default=0, help="0 runs until the window closes or Ctrl-C")
parser.add_argument(
    "--trace", type=Path, help="Save joint targets and measured poses for a finite run"
)
args = parser.parse_args()
assert args.robot_usd.is_file(), f"missing SO101 USD: {args.robot_usd}"
assert np.isfinite(args.amplitude) and args.amplitude > 0, "amplitude must be positive degrees"
assert np.isfinite(args.period) and args.period > 0, "period must be positive seconds"
assert args.steps >= 0 and args.gpu >= 0, "steps and gpu must be nonnegative"
assert args.trace is None or args.steps > 0, "--trace requires a finite --steps run"
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "active_gpu": args.gpu,
        "physics_gpu": args.gpu,
        "multi_gpu": False,
        "limit_cpu_threads": 16,
        "enable_crashreporter": False,
        "extra_args": ["--/telemetry/enableAnonymousData=false"],
    }
)

# Isaac's plugin modules are available only after SimulationApp starts.
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from pxr import PhysxSchema, UsdLux, UsdPhysics

DT = 1 / 120
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def main() -> None:  # 仿真入口，不返回结果；可选的运动记录会写入文件。
    # 创建仿真管理器：物理和渲染步长均为 DT=1/120 秒，场景长度单位为米。
    world = World(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    # 创建带碰撞的地面；size 指定地面可视几何的尺寸，此处为 2 米。
    world.scene.add_ground_plane(size=2.0)
    # 获取当前 USD 场景；后续通过它查找节点、编辑灯光和物理配置。
    stage = get_current_stage()
    # 在 /World/Light 定义穹顶环境光，并将其强度属性设置为 1000。
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000)
    # 将命令行指定的 USD 文件解析为绝对路径，以引用方式挂载到 /World/SO101。
    add_reference_to_stage(str(args.robot_usd.resolve()), "/World/SO101")

    # 获取资产已有的 root_joint 通用节点句柄；这里不会创建节点。
    root = stage.GetPrimAtPath("/World/SO101/root_joint")
    # 要求节点存在，并带有关节系统根标记；否则抛出下方消息对应的 AssertionError。
    assert root.IsValid() and root.HasAPI(UsdPhysics.ArticulationRootAPI), (
        "expected the official RobotStudio/so101_new_calib PhysX asset with root_joint"
    )
    # 为同一节点构造固定关节访问接口，不创建关节，也不修改节点类型。
    joint = UsdPhysics.FixedJoint(root)
    # 要求类型为固定关节、body0 目标为空（连接静态世界）、body1 目标非空。
    # 官方资产的 body1 指向底座；此处只检查目标列表非空，未核对具体路径。
    assert joint and not joint.GetBody0Rel().GetTargets() and joint.GetBody1Rel().GetTargets(), (
        "SO101 root_joint must fix its base to the world"
    )
    # 应用 PhysX 关节系统配置 schema，并写入关闭自碰撞的属性；这会修改场景配置。
    PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr(False)
    # 包装已有的整套机器人关节系统，并以 so101 为名称注册，供 world 初始化和重置。
    robot = world.scene.add(Articulation("/World/SO101/root_joint", name="so101"))
    # 设置视口相机：eye 是相机位置，target 是看向的点，均为世界坐标、单位米。
    set_camera_view(eye=np.array([0.65, -0.75, 0.55]), target=np.array([0.0, -0.12, 0.20]))
    # 初始化物理和已注册对象的运行时句柄，并重置默认状态；之后才能读写关节。
    world.reset()

    # 读取模拟器实际采用的关节顺序；后续数组的列必须按这个顺序排列。
    names = robot.dof_names
    # 要求恰好有预期的六个关节；集合比较允许资产中的关节排列顺序不同。
    assert len(names) == 6 and set(names) == set(JOINTS), f"unexpected SO101 joints: {names}"
    # 查出肩部水平转动关节和夹爪关节的列索引，避免硬编码关节下标。
    pan, gripper = names.index("shoulder_pan"), names.index("gripper")
    # 创建全零参考姿态，形状 (1, 6)：1 台机器人、6 个关节；此模型转角单位为弧度。
    home = np.zeros((1, len(names)), dtype=np.float32)
    # 将命令行输入的摆动幅度从度转换为弧度，与关节接口的单位一致。
    amplitude = np.deg2rad(args.amplitude)
    # 取第 0 台机器人的限位，得到 (6, 2) 数组；每行分别为该关节的下限、上限。
    limits = robot.get_dof_limits()[0]
    # 检查零位是否位于所有关节的允许范围内；失败时输出完整限位数组。
    assert ((home[0] >= limits[:, 0]) & (home[0] <= limits[:, 1])).all(), limits
    # 检查肩部目标的整个 [-amplitude, +amplitude] 范围严格位于限位之内。
    assert limits[pan, 0] < -amplitude and amplitude < limits[pan, 1], (
        f"sweep ±{args.amplitude} degrees exceeds shoulder_pan limits {np.rad2deg(limits[pan])}"
    )
    # 创建 (1, 6) 的 PD 增益数组：Kp=400、Kd=80；这些是仿真参数，非真机标定值。
    stiffness, damping = np.full_like(home, 400.0), np.full_like(home, 80.0)
    # 单独覆盖夹爪的增益，其余五个关节继续使用上面的值。
    stiffness[0, gripper], damping[0, gripper] = 17.8, 0.60
    # 将比例增益和微分增益写入各关节驱动器，决定位置、速度误差产生的驱动力矩。
    robot.set_gains(kps=stiffness, kds=damping)
    # 初始化时直接把实际关节角设为零位；这不是等待驱动器运动到零位。
    robot.set_joint_positions(home)
    # 同时将实际关节速度设为零，避免初始速度影响随后的运动。
    robot.set_joint_velocities(np.zeros_like(home))
    # 将驱动器的位置目标也设为零位，使初始目标与实际关节角一致。
    robot.set_joint_position_targets(home)
    # 输出关节顺序、摆幅和周期；flush=True 让重定向的日志也及时可见。
    print(f"SO101 joints: {names}; sweep ±{args.amplitude}°, period {args.period}s", flush=True)

    # 从第 0 个控制步开始计数；step * DT 是本步生成目标所用的仿真时间。
    step = 0
    # 创建轨迹缓存；只有提供 --trace 时才会向其中追加每步数据。
    samples = []
    # 应用未关闭且未达到步数上限时继续；--steps 0 表示不限制运行步数。
    while simulation_app.is_running() and (args.steps == 0 or step < args.steps):
        # 检查时间线是否在播放；暂停时不推进控制目标、物理时间和 step。
        if not world.is_playing():
            # 暂停期间仍刷新画面，让窗口保持响应。
            world.render()
            # 跳回循环开头，等待恢复播放或应用退出。
            continue
        # 复制零位作为本步目标；修改副本不会改变 home，其余关节保持零位目标。
        target = home.copy()
        # 只让肩部按 A*sin(2πt/T) 摆动：A 为弧度幅值，t=step*DT，T 为周期。
        target[0, pan] = amplitude * np.sin(2 * np.pi * step * DT / args.period)
        # 提交本步位置目标，由 PD 驱动器跟踪；这里不直接改变实际关节角。
        robot.set_joint_position_targets(target)
        # 推进一步物理仿真并渲染；本例物理和渲染步长相同。
        world.step(render=True)
        # 读取推进后的实际关节角，形状为 (1, 6)；实际值可能滞后于目标。
        measured = robot.get_joint_positions()
        # 任一关节角出现 NaN 或无穷大时立即失败，并报告控制步编号。
        assert np.isfinite(measured).all(), f"non-finite joint positions at step {step}"
        # 仅在指定轨迹输出文件时采样，避免无限运行时持续占用轨迹缓存。
        if args.trace is not None:
            # 读取机器人基座的世界位置和旋转；形状分别为 (1, 3)、(1, 4)。
            # 位置单位为米，旋转使用四元数，分量顺序为 (w, x, y, z)。
            base_position, base_rotation = robot.get_world_poses()
            # 将本步的四项数据作为一个元组保存；copy 避免后续缓冲区更新改写历史。
            samples.append(
                (
                    target[0].copy(),  # 取第 0 台机器人的六个目标关节角。
                    measured[0].copy(),  # 保存对应的六个实际关节角。
                    base_position[0].copy(),  # 保存基座三维位置，供检查是否漂移。
                    base_rotation[0].copy(),  # 保存基座朝向，供检查是否发生旋转。
                )
            )
        # 每隔 120 步打印一次，即目标时间每隔 1 秒打印一次；第 0 步也会打印。
        if step % 120 == 0:
            # 显示本步目标时间和步进后的实测肩部角度，角度从弧度转换为度。
            print(f"t={step * DT:.1f}s pan={np.rad2deg(measured[0, pan]):.1f}°", flush=True)
        # 本步完成后递增计数，下一步据此生成新的目标。
        step += 1

    # 循环退出后，只有指定 --trace 才保存轨迹；前面的参数检查已要求有限步数。
    if args.trace is not None:
        # 要求完整跑完指定步数；如果窗口提前关闭，则报错，不将残缺轨迹当作完整结果。
        assert step == args.steps, f"simulation closed early: {step}/{args.steps} steps"
        # 创建输出文件的父目录，允许目录已经存在，并自动补齐缺失的上级目录。
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        # 将按时间排列的四元组拆成四列，并分别转换成 NumPy 数组。
        # 结果形状依次为 (N, 6)、(N, 6)、(N, 3)、(N, 4)，N 是完成的步数。
        targets, positions, base_positions, base_rotations = map(
            np.asarray, zip(*samples, strict=True)  # strict=True 要求每条采样的字段数相同。
        )
        # 将轨迹和解释它所需的关节顺序、时间步长保存为压缩 NPZ 文件。
        np.savez_compressed(
            args.trace,  # --trace 指定的输出路径。
            joint_names=names,  # 保存关节列名，读取时无需猜测列顺序。
            targets=targets,  # 每步发给驱动器的目标关节角，单位弧度。
            positions=positions,  # 每步物理推进后的实测关节角，单位弧度。
            base_positions=base_positions,  # 每步基座的世界位置，单位米。
            base_rotations=base_rotations,  # 每步基座的世界朝向四元数。
            dt=DT,  # 相邻控制步的仿真时间间隔，单位秒。
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
