from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

from carrot_sim.arena_libero.mdp import tcp_state
from carrot_sim.arena_libero.predicates import (
    inside_drawer_and_closed,
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
) -> torch.Tensor:
    target = env.scene[target_cfg.name].data
    support = env.scene[support_cfg.name].data
    robot = env.scene[robot_cfg.name].data
    hand = robot_cfg.body_ids[0]
    tcp, _ = combine_frame_transforms(
        robot.body_pos_w.torch[:, hand],
        robot.body_quat_w.torch[:, hand],
        target.root_pos_w.torch.new_tensor([0.0, 0.0, 0.107]).expand(env.num_envs, -1),
    )
    released = released_and_stable(target.root_pos_w.torch, tcp, target.root_lin_vel_w.torch)
    if goal["support_body"]:
        body = support_cfg.body_ids[0]
        support_pos = support.body_pos_w.torch[:, body]
        support_quat = support.body_quat_w.torch[:, body]
    else:
        support_pos, support_quat = support.root_pos_w.torch, support.root_quat_w.torch
    local_pos, local_quat = subtract_frame_transforms(
        support_pos, support_quat, target.root_pos_w.torch, target.root_quat_w.torch
    )
    local_pos = local_pos - local_pos.new_tensor(goal["center"])
    if goal["kind"] == "in_closed_drawer":
        # Rotate the object's bounds into the moving drawer frame before testing containment.
        basis = torch.eye(3, device=env.device).expand(env.num_envs, -1, -1)
        rotation = quat_apply(local_quat[:, None].expand(-1, 3, -1), basis)
        extent = (
            rotation.abs() * local_pos.new_tensor(goal["target_half_size"])[None, :, None]
        ).sum(1)
        return released & inside_drawer_and_closed(
            local_pos,
            extent,
            local_pos.new_tensor(goal["half_size"]),
            support.joint_pos.torch[:, support_cfg.joint_ids][:, 0],
            goal["closed_position"],
            goal["closed_tolerance"],
        )
    force = env.scene["target_contact"].data.force_matrix_w.torch[:, 0, 0]
    placed = supported_on(local_pos, local_pos.new_tensor(goal["half_size"]), force)
    if goal["kind"] == "on_plate":
        placed &= local_pos[:, :2].norm(dim=-1) < goal["half_size"][0]
    if goal["kind"] == "on_lit_stove":
        placed &= stove_is_on(support.joint_pos.torch[:, support_cfg.joint_ids][:, 0])
    return released & placed


def dropped(env: ManagerBasedRLEnv, target_cfg: SceneEntityCfg) -> torch.Tensor:
    return (
        env.scene[target_cfg.name].data.root_pos_w.torch[:, 2] < env.scene.env_origins[:, 2] + 0.65
    )


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
