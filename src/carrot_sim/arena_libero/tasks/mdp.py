from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

from carrot_sim.arena_libero.mdp import tcp_state
from carrot_sim.arena_libero.predicates import (
    inside_drawer_and_closed,
    placed_beside,
    released_and_stable,
    stove_is_on,
    supported_on,
)


def success(
    env: ManagerBasedRLEnv,
    goal: dict[str, Any],
    target_cfg: SceneEntityCfg,
    support_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    contact_name: str,
) -> torch.Tensor:
    support = env.scene[support_cfg.name].data
    robot = env.scene[robot_cfg.name].data
    hand = robot_cfg.body_ids[0]
    tcp, _ = combine_frame_transforms(
        robot.body_pos_w.torch[:, hand],
        robot.body_quat_w.torch[:, hand],
        robot.root_pos_w.torch.new_tensor([0.0, 0.0, 0.107]).expand(env.num_envs, -1),
    )
    if goal["support_body"]:
        support_pos = support.body_pos_w.torch[:, support_cfg.body_ids][:, 0]
        support_quat = support.body_quat_w.torch[:, support_cfg.body_ids][:, 0]
    else:
        support_pos, support_quat = support.root_pos_w.torch, support.root_quat_w.torch
    if goal["kind"].startswith("joint_"):
        joint = support.joint_pos.torch[:, support_cfg.joint_ids][:, 0]
        anchor, _ = combine_frame_transforms(
            support_pos,
            support_quat,
            support_pos.new_tensor(goal["center"]).expand(env.num_envs, -1),
        )
        released = (tcp - anchor).norm(dim=-1) > goal["release_distance"]
        if goal["kind"] == "joint_open":
            return released & (joint >= goal["joint_min"]) & (joint <= goal["joint_max"])
        if goal["kind"] == "joint_closed":
            return released & ((joint - goal["closed_position"]).abs() <= goal["closed_tolerance"])
        on = stove_is_on(joint)
        return released & (on if goal["kind"] == "joint_on" else ~on)
    target = env.scene[target_cfg.name].data
    released = released_and_stable(
        target.root_pos_w.torch,
        tcp,
        target.root_lin_vel_w.torch,
        goal["release_distance"],
    )
    up = quat_apply(
        target.root_quat_w.torch,
        target.root_pos_w.torch.new_tensor([0.0, 0.0, 1.0]).expand(env.num_envs, -1),
    )
    released &= up[:, 2] >= goal["min_upright_cos"]
    geometry_center, _ = combine_frame_transforms(
        target.root_pos_w.torch,
        target.root_quat_w.torch,
        target.root_pos_w.torch.new_tensor(goal["target_center"]).expand(env.num_envs, -1),
    )
    local_pos, local_quat = subtract_frame_transforms(
        support_pos,
        support_quat,
        geometry_center,
        target.root_quat_w.torch,
    )
    if goal["frame"] == "world":
        local_pos = geometry_center - support_pos
        local_quat = target.root_quat_w.torch
    local_pos = local_pos - local_pos.new_tensor(goal["center"])
    if goal["kind"] in ("inside", "in_closed_drawer"):
        # Bound the whole rotated object, not only its center, inside the moving receptacle.
        basis = torch.eye(3, device=env.device).expand(env.num_envs, -1, -1)
        rotation = quat_apply(local_quat[:, None].expand(-1, 3, -1), basis)
        extent = (
            rotation.abs() * local_pos.new_tensor(goal["target_half_size"])[None, :, None]
        ).sum(1)
        half = local_pos.new_tensor(goal["half_size"])
        inside = ((local_pos.abs() + extent) < half).all(-1)
        if goal["containment"] == "partial":
            overlap = (
                torch.minimum(local_pos + extent, half) - torch.maximum(local_pos - extent, -half)
            ).clamp_min(0)
            fraction = overlap.prod(-1) / (2 * extent).prod(-1).clamp_min(1e-9)
            bottom = local_pos[:, 2] - extent[:, 2]
            inside = (fraction >= goal["overlap_fraction"]) & ((bottom + half[2]).abs() < 0.015)
        if goal["containment"] == "opening":
            bottom = local_pos[:, 2] - extent[:, 2]
            force = env.scene[contact_name].data.force_matrix_w.torch[:, 0, 0]
            inside = ((local_pos[:, :2].abs() + extent[:, :2]) < half[:2]).all(-1)
            inside &= (bottom > -half[2] - 0.01) & (bottom < half[2]) & (force[:, 2] > 0.1)
        if goal["kind"] == "in_closed_drawer":
            inside = inside_drawer_and_closed(
                local_pos,
                extent,
                half,
                support.joint_pos.torch[:, support_cfg.joint_ids][:, 0],
                goal["closed_position"],
                goal["closed_tolerance"],
            )
        if goal["require_contact"]:
            force = env.scene[contact_name].data.force_matrix_w.torch[:, 0, 0]
            inside &= force[:, 2] > 0.1
        return released & inside
    if goal["kind"] == "relative":
        return released & (local_pos.abs() < local_pos.new_tensor(goal["half_size"])).all(-1)
    if goal["kind"] == "beside":
        basis = torch.eye(3, device=env.device).expand(env.num_envs, -1, -1)
        target_rotation = quat_apply(target.root_quat_w.torch[:, None].expand(-1, 3, -1), basis)
        support_rotation = quat_apply(support_quat[:, None].expand(-1, 3, -1), basis)
        target_extent = (
            target_rotation.abs() * local_pos.new_tensor(goal["target_half_size"])[None, :, None]
        ).sum(1)
        support_extent = (
            support_rotation.abs() * local_pos.new_tensor(goal["support_half_size"])[None, :, None]
        ).sum(1)
        support_center, _ = combine_frame_transforms(
            support_pos,
            support_quat,
            support_pos.new_tensor(goal["support_center"]).expand(env.num_envs, -1),
        )
        return released & placed_beside(
            geometry_center - support_center,
            target_extent,
            support_extent,
            goal["side"],
            goal["edge_distance"],
            goal["side_tolerance"],
        )
    force = env.scene[contact_name].data.force_matrix_w.torch[:, 0, 0]
    placed = supported_on(local_pos, local_pos.new_tensor(goal["half_size"]), force)
    if goal["kind"] == "on_plate":
        placed &= local_pos[:, :2].norm(dim=-1) < goal["half_size"][0]
    if goal["kind"] == "on_lit_stove":
        placed &= stove_is_on(support.joint_pos.torch[:, support_cfg.joint_ids][:, 0])
    return released & placed


def all_success(
    env: ManagerBasedRLEnv,
    goals: tuple[dict[str, Any], ...],
    targets: tuple[SceneEntityCfg, ...],
    supports: tuple[SceneEntityCfg, ...],
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    flags = [
        success(env, goal, target, support, robot_cfg, f"target_contact_{index}")
        for index, (goal, target, support) in enumerate(zip(goals, targets, supports, strict=True))
    ]
    return torch.stack(flags).all(dim=0)


def dropped(env: ManagerBasedRLEnv, targets: tuple[str, ...]) -> torch.Tensor:
    flags = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for name in targets:
        flags |= env.scene[name].data.root_pos_w.torch[:, 2] < env.scene.env_origins[:, 2] + 0.65
    return flags


def critic_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    entities: tuple[SceneEntityCfg, ...],
    fixtures: tuple[SceneEntityCfg, ...],
) -> torch.Tensor:
    robot = env.scene["robot"].data
    parts = [tcp_state(env, robot_cfg), robot.joint_pos.torch, robot.joint_vel.torch]
    for cfg in entities:
        data = env.scene[cfg.name].data
        pos, quat = subtract_frame_transforms(
            robot.root_pos_w.torch,
            robot.root_quat_w.torch,
            data.root_pos_w.torch,
            data.root_quat_w.torch,
        )
        parts.extend((pos, quat, data.root_lin_vel_w.torch, data.root_ang_vel_w.torch))
    for cfg in fixtures:
        data = env.scene[cfg.name].data
        parts.extend(
            (data.joint_pos.torch[:, cfg.joint_ids], data.joint_vel.torch[:, cfg.joint_ids])
        )
    return torch.cat(parts, dim=-1)
