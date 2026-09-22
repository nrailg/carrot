from collections import Counter, defaultdict
from math import cos, radians, sin, sqrt

import torch
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_mul

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
        if goal.containment in ("opening", "partial"):
            offset[:, 2] += -goal.half_size[2] + extent[2] + 0.005
        if goal.containment == "partial":
            # 开放式柜子允许部分插入：下层插入薄柄，中层放锅体并让柄留在柜外。
            if goal.center[2] < 0:
                rotation = pos.new_tensor([[0.0, 0.0, 1.0, 0.0]])
                offset[:, 1] -= goal.half_size[1] + target_half[1] - 0.075
            else:
                rotation = pos.new_tensor([[0.0, 0.0, 0.0, 1.0]])
                offset[:, 1] -= target_half[1] * 0.6
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


def success_fixture(env, spec: TaskSpec) -> None:
    # 只构造 env 0 的物理成功状态，不能将其结果广播到其他 slot。
    env.reset(seed=42)
    backend = env.backend
    old_duration = backend.cfg.episode_length_s
    backend.cfg.episode_length_s = 6.0
    ids = torch.tensor([0], device=env.device, dtype=torch.int32)
    neutral = torch.zeros(env.num_envs, 7, device=env.device)
    neutral[:, 6] = -1
    try:
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
        for _ in range(120):
            _, reward, term, _, info = env.step(neutral)
            assert not info["success"][1:].any(), "success leaked to another environment"
            if info["success"][0]:
                assert term[0] and reward[0] > 0 and not info["bootstrap_mask"][0]
                assert info["final_observation"] is not None
                return
        raise AssertionError(f"Physical success fixture failed: {spec.task_id}")
    finally:
        backend.cfg.episode_length_s = old_duration
