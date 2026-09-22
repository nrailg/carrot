import json
from collections import Counter, defaultdict
from dataclasses import asdict
from math import cos, radians, sin, sqrt
from pathlib import Path

import torch
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_mul,
    subtract_frame_transforms,
)
from PIL import Image

from carrot_sim.arena_libero.tasks.spec import GoalSpec, TaskSpec


def set_goal_joint(env, goal: GoalSpec, value: float) -> None:
    support = env.backend.scene[goal.support]
    ids = torch.tensor([0], device=env.device, dtype=torch.int32)
    position = support.data.joint_pos.torch[ids].clone()
    position[:, support.find_joints(goal.joint)[0][0]] = value
    support.write_joint_position_to_sim_index(position=position, env_ids=ids)
    support.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(position), env_ids=ids)
    env.backend.sim.forward()
    env.backend.scene.update(env.backend.physics_dt)


def support_frame(env, goal: GoalSpec) -> tuple[torch.Tensor, torch.Tensor]:
    support = env.backend.scene[goal.support]
    if goal.support_body:
        body = support.find_bodies(goal.support_body)[0][0]
        return support.data.body_pos_w.torch[:1, body], support.data.body_quat_w.torch[:1, body]
    return support.data.root_pos_w.torch[:1], support.data.root_quat_w.torch[:1]


def placement_pose(
    env, goal: GoalSpec, group_size: int, group_index: int, support_asset: str
) -> tuple[torch.Tensor, torch.Tensor]:
    pos, quat = support_frame(env, goal)
    offset = pos.new_tensor(goal.center).reshape(1, 3)
    target_half = pos.new_tensor(goal.target_half_size)
    rotation = pos.new_tensor([[0.0, 0.0, 0.0, 1.0]])
    extent = target_half
    if goal.kind in ("inside", "in_closed_drawer"):
        # 从几种正交朝向选能放入槽位的一种，酒瓶/书不必强行直立。
        s = sqrt(0.5)
        candidates = pos.new_tensor(
            [
                [0, 0, 0, 1],
                [0, 0, s, s],
                [s, 0, 0, s],
                [0, s, 0, s],
                [0.5, 0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5, 0.5],
            ]
        )
        axes = torch.eye(3, device=env.device).expand(len(candidates), -1, -1)
        extents = (
            quat_apply(candidates[:, None].expand(-1, 3, -1), axes).abs()
            * target_half[None, :, None]
        ).sum(1)
        available = pos.new_tensor(goal.half_size)
        available[0] /= group_size
        if goal.containment in ("opening", "partial"):
            available[2] = float("inf")
        fits = (extents < available - 0.002).all(-1)
        if goal.containment == "partial":
            fits.zero_()
        if fits.any():
            selected = int(fits.nonzero()[0, 0])
            rotation, extent = candidates[selected : selected + 1], extents[selected]
        elif goal.containment != "partial":
            raise AssertionError(f"No fixture orientation fits {goal.target} in {goal.support}")
        if goal.containment in ("opening", "partial") or support_asset == "microwave":
            offset[:, 2] += -goal.half_size[2] + extent[2] + 0.005
        if goal.containment == "partial":
            # 开放式柜子允许部分插入：下层插入薄柄，中层放锅体并让柄留在柜外。
            if goal.center[2] < 0:
                rotation = pos.new_tensor([[0.0, 0.0, 1.0, 0.0]])
                offset[:, 1] -= goal.half_size[1] + target_half[1] - 0.075
            else:
                rotation = pos.new_tensor([[0.0, 0.0, 0.0, 1.0]])
                offset[:, 1] -= target_half[1] * 0.6
                # 中板只有 3.45 mm 厚，夹具从靠近板面处释放，避免额外自由落体。
                offset[:, 2] -= 0.005
    elif goal.kind == "beside":
        axis = 0 if goal.side in ("left", "right") else 1
        sign = 1 if goal.side in ("left", "back") else -1
        axes = torch.eye(3, device=env.device).reshape(1, 3, 3)
        support_extent = (
            quat_apply(quat[:, None].expand(-1, 3, -1), axes).abs()
            * pos.new_tensor(goal.support_half_size)[None, :, None]
        ).sum(1)[0]
        distance = target_half[axis] + support_extent[axis] + sum(goal.edge_distance) / 2
        pos, _ = combine_frame_transforms(
            pos, quat, pos.new_tensor(goal.support_center).reshape(1, 3)
        )
        offset.zero_()
        offset[:, axis] = sign * distance
        offset[:, 2] = 0.755 + target_half[2] - pos[:, 2]
        quat = pos.new_tensor([[0.0, 0.0, 0.0, 1.0]])
    elif goal.kind != "relative":
        offset[:, 2] += target_half[2] + 0.018
    if support_asset == "microwave":
        # 微波炉内从接近转盘处释放；内腔底部的 2 mm 余量不作为落体空间。
        offset[:, 2] -= 0.002
    if support_asset == "wine_rack":
        # 酒瓶沿真实斜板横放并避开中央立柱，再由物理接触决定是否稳定。
        angle = radians(65) / 2
        rotation = pos.new_tensor([[-sin(angle), 0.0, 0.0, cos(angle)]])
        offset = pos.new_tensor([[0.08, -0.01, 0.13]])
    if group_size > 1:
        # 同一容器的多个目标分配不同槽位，不能互相穿插后直接判成功。
        slot_width = 2 * goal.half_size[0] / group_size
        offset[:, 0] += (group_index - (group_size - 1) / 2) * slot_width
    if goal.frame == "world":
        quat = pos.new_tensor([[0.0, 0.0, 0.0, 1.0]])
    geometry_pos, _ = combine_frame_transforms(pos, quat, offset)
    target_quat = quat_mul(quat, rotation)
    target_pos = geometry_pos - quat_apply(
        target_quat, pos.new_tensor(goal.target_center).reshape(1, 3)
    )
    return target_pos, target_quat


def _rotated_extent(half_size: tuple[float, float, float], quat: torch.Tensor) -> torch.Tensor:
    axes = torch.eye(3, device=quat.device, dtype=quat.dtype).expand(quat.shape[0], -1, -1)
    rotated_axes = quat_apply(quat[:, None].expand(-1, 3, -1), axes)
    return (rotated_axes.abs() * quat.new_tensor(half_size)[None, :, None]).sum(1)


def _geometry_diagnostics(
    goal: GoalSpec,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    support_pos: torch.Tensor,
    support_quat: torch.Tensor,
) -> dict[str, object]:
    geometry_center, _ = combine_frame_transforms(
        target_pos, target_quat, target_pos.new_tensor(goal.target_center).reshape(1, 3)
    )
    local_pos, local_quat = subtract_frame_transforms(
        support_pos, support_quat, geometry_center, target_quat
    )
    goal_pos = geometry_center - support_pos if goal.frame == "world" else local_pos
    goal_quat = target_quat if goal.frame == "world" else local_quat
    offset = goal_pos - goal_pos.new_tensor(goal.center)
    extent = _rotated_extent(goal.target_half_size, goal_quat)
    half = offset.new_tensor(goal.half_size)
    overlap = (
        torch.minimum(offset + extent, half) - torch.maximum(offset - extent, -half)
    ).clamp_min(0)
    bottom = offset[:, 2] - extent[:, 2]
    support_center, _ = combine_frame_transforms(
        support_pos, support_quat, support_pos.new_tensor(goal.support_center).reshape(1, 3)
    )
    return {
        "target_geometry_center_w": geometry_center[0].tolist(),
        "target_geometry_center_support_frame": local_pos[0].tolist(),
        "target_geometry_offset_goal_frame": offset[0].tolist(),
        "target_quat_goal_frame_xyzw": goal_quat[0].tolist(),
        "rotated_half_size_goal_frame": extent[0].tolist(),
        "full_containment_margin": (half - offset.abs() - extent)[0].tolist(),
        "target_bottom_goal_frame": bottom[0].item(),
        "opening_bottom_lower_margin": (bottom + half[2] + 0.01)[0].item(),
        "opening_bottom_upper_margin": (half[2] - bottom)[0].item(),
        "partial_overlap_fraction": (overlap.prod(-1) / (2 * extent).prod(-1).clamp_min(1e-9))[
            0
        ].item(),
        "support_geometry_center_w": support_center[0].tolist(),
        "target_geometry_delta_support_w": (geometry_center - support_center)[0].tolist(),
        "target_half_size_w": _rotated_extent(goal.target_half_size, target_quat)[0].tolist(),
        "support_half_size_w": _rotated_extent(goal.support_half_size, support_quat)[0].tolist(),
    }


def _goal_diagnostics(env, spec: TaskSpec, index: int, tcp: torch.Tensor) -> dict[str, object]:
    goal = spec.conditions[index]
    support = env.backend.scene[goal.support]
    support_pos, support_quat = support_frame(env, goal)
    anchor, _ = combine_frame_transforms(
        support_pos, support_quat, support_pos.new_tensor(goal.center).reshape(1, 3)
    )
    support_spec = next(
        item for item in (*spec.objects, *spec.fixtures) if item.name == goal.support
    )
    diagnostic: dict[str, object] = {
        "index": index,
        "goal": asdict(goal),
        "support_root_pose_w_xyzw": support.data.root_pose_w.torch[0].tolist(),
        "support_body_pose_w_xyzw": torch.cat((support_pos, support_quat), -1)[0].tolist()
        if goal.support_body
        else None,
        "support_joints": {
            name: support.data.joint_pos.torch[0, support.find_joints(name)[0][0]].item()
            for name, _ in support_spec.joints
        },
        "support_bodies": {
            name: support.data.body_pose_w.torch[0, body].tolist()
            for body, name in enumerate(support.body_names)
        }
        if support_spec.joints
        else {},
        "tcp_distance_to_goal_anchor": (tcp - anchor).norm(dim=-1)[0].item(),
        "contact_sensor": None,
        "contact_force_w": None,
        "contact_force_z_threshold": 0.1,
        "target": None,
    }
    if goal.target:
        target = env.backend.scene[goal.target]
        target_pos = target.data.root_pos_w.torch[:1]
        target_quat = target.data.root_quat_w.torch[:1]
        target_spec = next(item for item in spec.objects if item.name == goal.target)
        up = quat_apply(target_quat, target_pos.new_tensor([[0.0, 0.0, 1.0]]))
        diagnostic["target"] = {
            "root_pose_w_xyzw": target.data.root_pose_w.torch[0].tolist(),
            "linear_velocity_w": target.data.root_lin_vel_w.torch[0].tolist(),
            "angular_velocity_w": target.data.root_ang_vel_w.torch[0].tolist(),
            "linear_speed": target.data.root_lin_vel_w.torch[0].norm().item(),
            "linear_speed_threshold": 0.05,
            "angular_speed": target.data.root_ang_vel_w.torch[0].norm().item(),
            "tcp_distance": (tcp - target_pos).norm(dim=-1)[0].item(),
            "up_cos": up[0, 2].item(),
            "joints": {
                name: target.data.joint_pos.torch[0, target.find_joints(name)[0][0]].item()
                for name, _ in target_spec.joints
            },
            **_geometry_diagnostics(goal, target_pos, target_quat, support_pos, support_quat),
        }
    if (
        goal.kind in ("on_plate", "on_lit_stove", "on_surface")
        or goal.containment == "opening"
        or goal.require_contact
    ):
        contact_name = f"target_contact_{index}"
        diagnostic["contact_sensor"] = contact_name
        diagnostic["contact_force_w"] = (
            env.backend.scene[contact_name].data.force_matrix_w.torch[0, 0, 0].tolist()
        )
    return diagnostic


def _failure_diagnostics(env, spec: TaskSpec) -> dict[str, object]:
    robot = env.backend.scene["robot"]
    hand = robot.find_bodies("panda_hand")[0][0]
    tcp, _ = combine_frame_transforms(
        robot.data.body_pos_w.torch[:1, hand],
        robot.data.body_quat_w.torch[:1, hand],
        robot.data.root_pos_w.torch.new_tensor([[0.0, 0.0, 0.107]]),
    )
    return {
        "event": "physical_success_fixture_failed",
        "task_id": spec.task_id,
        "env_id": 0,
        "steps": 120,
        "tcp_position_w": tcp[0].tolist(),
        "goals": [
            _goal_diagnostics(env, spec, index, tcp) for index in range(len(spec.conditions))
        ],
    }


def success_fixture(env, spec: TaskSpec, output: Path) -> None:
    # 只构造 env 0 的物理成功状态，不能将其结果广播到其他 slot。
    env.reset(seed=42)
    backend = env.backend
    old_duration = backend.cfg.episode_length_s
    backend.cfg.episode_length_s = 6.0
    ids = torch.tensor([0], device=env.device, dtype=torch.int32)
    neutral = torch.zeros(env.num_envs, 7, device=env.device)
    neutral[:, 6] = -1
    try:
        # 先用真实控制将 env 0 的手退向基座并抬高，满足任务各自的释放距离。
        # Panda 基座绕 Z 旋转 -90 度，局部 -X 对应世界 +Y；其余环境保持零动作。
        retreat = neutral.clone()
        retreat[0, 0] = -0.5
        retreat[0, 2] = 0.5
        for _ in range(40):
            env.step(retreat)
        for goal in spec.conditions:
            if not goal.joint:
                continue
            if goal.kind in ("in_closed_drawer", "joint_closed", "joint_off"):
                value = goal.closed_position
            elif goal.kind == "joint_open":
                value = (goal.joint_min + goal.joint_max) / 2
            else:
                value = 0.8
            set_goal_joint(env, goal, value)
        if any(goal.joint for goal in spec.conditions) and any(
            goal.target for goal in spec.conditions
        ):
            # 让关节瞬移后的刚体/碰撞状态经过真实物理步，再向容器摆入目标。
            for _ in range(5):
                env.step(neutral)

        goals = [goal for goal in spec.conditions if goal.target]
        groups = Counter(goal.support for goal in goals if goal.kind not in ("beside", "relative"))
        assigned = defaultdict(int)
        pending = list(goals)
        while pending:
            # 先放承载下一个目标的物体，例如先放托盘里的下碗，再叠上碗。
            movable_supports = {goal.target for goal in pending}
            ready = [goal for goal in pending if goal.support not in movable_supports]
            if not ready:
                raise AssertionError("Cyclic goal supports")
            for goal in ready:
                packed = goal.kind not in ("beside", "relative")
                position, quaternion = placement_pose(
                    env,
                    goal,
                    groups[goal.support] if packed else 1,
                    assigned[goal.support] if packed else 0,
                    next(
                        item.asset
                        for item in (*spec.objects, *spec.fixtures)
                        if item.name == goal.support
                    ),
                )
                if packed:
                    assigned[goal.support] += 1
                target = backend.scene[goal.target]
                target.write_root_pose_to_sim_index(
                    root_pose=torch.cat((position, quaternion), -1), env_ids=ids
                )
                target.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids
                )
                pending.remove(goal)
            backend.sim.forward()
            backend.scene.update(backend.physics_dt)

        # 接触和静止必须经真实 PhysX 达成，成功还必须正确触发 terminal 接口。
        # 保留少量沉降轨迹，失败时区分初始穿插、滑落和静止后判据拒绝。
        history = [{"step": 0, **_failure_diagnostics(env, spec)}]
        images = {}
        for step in range(120):
            observation, reward, term, _, info = env.step(neutral)
            assert not info["success"][1:].any(), "success leaked to another environment"
            if info["success"][0]:
                assert term[0] and reward[0] > 0 and not info["bootstrap_mask"][0]
                assert info["final_observation"] is not None
                return
            if step + 1 in (1, 5, 15, 120):
                images[step + 1] = observation["image"][0].cpu().numpy().copy()
            if step + 1 in (1, 5, 15, 30, 60):
                history.append({"step": step + 1, **_failure_diagnostics(env, spec)})
        diagnostic = _failure_diagnostics(env, spec)
        diagnostic["settling_history"] = history
        (output / "physical_failure.json").write_text(json.dumps(diagnostic, indent=2))
        for step, pixels in images.items():
            Image.fromarray(pixels).save(output / f"fixture_step_{step}.png")
        print(json.dumps(diagnostic, indent=2), flush=True)
        raise AssertionError(f"Physical success fixture failed: {spec.task_id}")
    finally:
        backend.cfg.episode_length_s = old_duration
