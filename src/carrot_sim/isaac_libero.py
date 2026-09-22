"""Import after AppLauncher, inside the pinned Lightwheel Isaac Sim environment."""

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    quat_apply_inverse,
    subtract_frame_transforms,
)
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
    IsaacLabArenaManagerBasedRLEnvCfg,
)
from lw_benchhub.utils.env import ExecuteMode, parse_env_cfg
from typing_extensions import override

from carrot_sim.libero import IsaacLiberoConfig, LiberoVectorEnv, Observation, clone_observation


def _tcp_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    pos, quat = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, body_id],
        robot.data.body_quat_w[:, body_id],
    )
    offset = pos.new_tensor([0.0, 0.0, 0.107]).expand(env.num_envs, -1)
    pos, quat = combine_frame_transforms(pos, quat, offset)
    return torch.cat(
        (pos, axis_angle_from_quat(quat), robot.data.joint_pos[:, asset_cfg.joint_ids]), dim=-1
    )


def _critic_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    parts = [_tcp_state(env, asset_cfg), robot.data.joint_pos, robot.data.joint_vel]
    for name in ("akita_black_bowl", "plate"):
        obj = env.scene[name]
        pos, quat = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            obj.data.root_pos_w,
            obj.data.root_quat_w,
        )
        parts.extend(
            (
                pos,
                quat,
                quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w),
                quat_apply_inverse(robot.data.root_quat_w, obj.data.root_ang_vel_w),
            )
        )
    return torch.cat(parts, dim=-1)


def _rgb(env: ManagerBasedRLEnv, camera: str) -> torch.Tensor:
    return env.scene[camera].data.output["rgb"][..., :3]


def _robot_selection() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        body_names=["panda_hand"],
        joint_names=["panda_finger_joint1", "panda_finger_joint2"],
        preserve_order=True,
    )


@configclass
class _PolicyObservations(ObservationGroupCfg):
    state = ObservationTermCfg(func=_tcp_state, params={"asset_cfg": _robot_selection()})
    image = ObservationTermCfg(func=_rgb, params={"camera": "base_camera"})
    wrist_image = ObservationTermCfg(func=_rgb, params={"camera": "wrist_camera"})
    concatenate_terms = False
    enable_corruption = False


@configclass
class _CriticObservations(ObservationGroupCfg):
    state = ObservationTermCfg(func=_critic_state, params={"asset_cfg": _robot_selection()})
    concatenate_terms = True
    enable_corruption = False


@configclass
class _Observations:
    policy = _PolicyObservations()
    critic = _CriticObservations()


class _ResetCaptureEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> None:
        self._in_step = False
        self._final_observation: Observation | None = None
        self._transition_flags: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        super().__init__(cfg)

    @override
    def _configure_gym_env_spaces(self) -> None:
        super()._configure_gym_env_spaces()
        # Isaac's generic space builder describes every observation as float32.
        size = self.cfg.scene.base_camera.height
        for name in ("image", "wrist_image"):
            self.single_observation_space["policy"][name] = gym.spaces.Box(
                0, 255, (size, size, 3), dtype=np.uint8
            )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

    @override
    def step(
        self, action: torch.Tensor
    ) -> tuple[Observation, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._final_observation = None
        self._transition_flags = None
        self._in_step = True
        try:
            observation, reward, terminated, truncated, _ = super().step(action)
        finally:
            self._in_step = False
        if self._transition_flags is None:
            success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            terminated, truncated, success = self._transition_flags
        return (
            observation,
            reward,
            terminated,
            truncated,
            {
                "success": success,
                "final_observation": self._final_observation,
            },
        )

    @override
    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        if self._in_step:
            # Isaac resets before returning step(); snapshot both sensors and flags here.
            self._final_observation = clone_observation(self.observation_manager.compute())
            self._transition_flags = (
                self.termination_manager.terminated.clone(),
                self.termination_manager.time_outs.clone(),
                self.termination_manager.get_term("success").clone(),
            )
        super()._reset_idx(env_ids)
        # LW keeps success history outside Isaac's managers; reset only the selected slots.
        task = self.cfg.isaaclab_arena_env.task
        task._success_cache[env_ids] = 0
        task._success_flag[env_ids] = False
        focus = (
            self.scene["akita_black_bowl"].data.root_pos_w[env_ids]
            + self.scene["plate"].data.root_pos_w[env_ids]
        ) * 0.5
        self.scene["base_camera"].set_world_poses_from_view(
            eyes=focus + focus.new_tensor([0.0, -0.5, 0.85]), targets=focus, env_ids=env_ids
        )


def make_isaac_libero_env(config: IsaacLiberoConfig) -> LiberoVectorEnv:
    """Build one GPU-batched Lightwheel LIBERO simulator after AppLauncher starts.

    Parameters
    ----------
    config : IsaacLiberoConfig
        Uses Lightwheel's pinned Isaac Sim 5.0 stack. AppLauncher must have
        ``enable_cameras=True``; the caller owns and closes the application.

    Returns
    -------
    LiberoVectorEnv
        Policy state ``[N,8]``, two uint8 RGB ``[N,H,W,3]`` images, privileged
        critic state ``[N,52]``, and normalized action ``[N,7]``.
    """
    cfg = parse_env_cfg(
        scene_backend="robocasa",
        task_backend="robocasa",
        scene_name="libero-1-1",
        robot_name="Panda",
        task_name=LiberoVectorEnv.task_name,
        robot_scale=1.0,
        execute_mode=ExecuteMode.EVAL,
        device=config.device,
        num_envs=config.num_envs,
        use_fabric=True,
        enable_cameras=True,
        headless_mode=True,
        seed=config.seed,
        sources=["objaverse", "lightwheel", "aigen_objs"],
        resample_objects_placement_on_reset=True,
        resample_robot_placement_on_reset=False,
    )
    cfg.seed = config.seed
    # LW's generic robot placement puts the base at floor height. Mount Panda
    # at this scene's counter height so its initial arm does not intersect it.
    base_pos = cfg.scene.robot.init_state.pos
    cfg.scene.robot.init_state.pos = (base_pos[0], base_pos[1], 0.75)
    cfg.observations = _Observations()
    cfg.actions.arm_action.controller.use_relative_mode = True
    cfg.actions.arm_action.scale = (config.translation_scale,) * 3 + (config.rotation_scale,) * 3
    cfg.scene.ee_frame.target_frames[0].offset.pos = (0.0, 0.0, 0.107)
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = config.max_episode_steps * cfg.sim.dt * cfg.decimation
    cfg.is_finite_horizon = False
    cfg.rerender_on_reset = True
    cfg.scene.base_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/base_camera",
        update_period=0.0,
        height=config.image_size,
        width=config.image_size,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 0.8), rot=(0.65328, 0.27060, 0.27060, 0.65328), convention="opengl"
        ),
    )
    cfg.scene.wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera",
        update_period=0.0,
        height=config.image_size,
        width=config.image_size,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(focal_length=12.0, clipping_range=(0.01, 20.0)),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.05),
            rot=(0.90777199, -0.00438587, -0.37390275, 0.19007240),
            convention="ros",
        ),
    )
    return LiberoVectorEnv(_ResetCaptureEnv(cfg))
