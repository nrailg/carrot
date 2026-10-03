import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    subtract_frame_transforms,
)

from carrot_sim.arena_libero.predicates import bowl_on_plate


def tcp_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    pos, quat = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        robot.data.body_pos_w.torch[:, asset_cfg.body_ids[0]],
        robot.data.body_quat_w.torch[:, asset_cfg.body_ids[0]],
    )
    pos, quat = combine_frame_transforms(
        pos, quat, pos.new_tensor([0.0, 0.0, 0.107]).expand(env.num_envs, -1)
    )
    return torch.cat(
        (pos, axis_angle_from_quat(quat), robot.data.joint_pos.torch[:, asset_cfg.joint_ids]),
        dim=-1,
    )


def critic_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene["robot"]
    parts = [tcp_state(env, asset_cfg), robot.data.joint_pos.torch, robot.data.joint_vel.torch]
    for name in ("bowl", "plate"):
        data = env.scene[name].data
        pos, quat = subtract_frame_transforms(
            robot.data.root_pos_w.torch,
            robot.data.root_quat_w.torch,
            data.root_pos_w.torch,
            data.root_quat_w.torch,
        )
        parts.extend((pos, quat, data.root_lin_vel_w.torch, data.root_ang_vel_w.torch))
    return torch.cat(parts, dim=-1)


def rgb(env: ManagerBasedRLEnv, camera: str) -> torch.Tensor:
    return env.scene[camera].data.output["rgb"].torch[..., :3]


def placed(env: ManagerBasedRLEnv) -> torch.Tensor:
    bowl, plate = env.scene["bowl"].data, env.scene["plate"].data
    delta = bowl.root_pos_w.torch - plate.root_pos_w.torch
    force = env.scene["bowl_contact"].data.force_matrix_w.torch[:, 0, 0]
    fingers = env.scene["robot"].data.joint_pos.torch[:, -2:]
    return bowl_on_plate(delta, force, bowl.root_lin_vel_w.torch, fingers)


def success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.termination_manager.get_term("success").float()


def dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["bowl"].data.root_pos_w.torch[:, 2] < env.scene.env_origins[:, 2] + 0.65
