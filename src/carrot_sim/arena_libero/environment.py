"""Isaac runtime: import only after AppLauncher has started."""

from collections.abc import Sequence
from typing import Any, override

import torch
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
    IsaacLabArenaManagerBasedRLEnv,
)
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
    IsaacLabArenaManagerBasedRLEnvCfg,
)

from carrot_sim.arena_libero.config import ArenaLiberoConfig
from carrot_sim.arena_libero.robot import Panda
from carrot_sim.arena_libero.scene import KitchenScene
from carrot_sim.arena_libero.task import BowlOnPlateTask
from carrot_sim.arena_libero.tasks import get_task
from carrot_sim.arena_libero.tasks.runtime import LiberoTask
from carrot_sim.arena_libero.tasks.scene import TaskScene
from carrot_sim.arena_libero.vector_env import ArenaLiberoEnv


class TransitionEnv(IsaacLabArenaManagerBasedRLEnv):
    def __init__(
        self,
        cfg: IsaacLabArenaManagerBasedRLEnvCfg,
        camera_eye: tuple[float, float, float] = (2.48, -2.65, 1.65),
        camera_target: tuple[float, float, float] = (2.48, -2.04, 0.80),
        **kwargs: Any,
    ) -> None:
        self._camera_eye = camera_eye
        self._camera_target = camera_target
        self._stepping = False
        self._final = None
        self._flags = None
        super().__init__(cfg, **kwargs)

    @override
    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        if self._stepping:
            obs = self.observation_manager.compute()
            self._final = {
                "policy": {name: value.clone() for name, value in obs["policy"].items()},
                "critic": obs["critic"].clone(),
            }
            self._flags = (
                self.termination_manager.terminated.clone(),
                self.termination_manager.time_outs.clone(),
                self.termination_manager.get_term("success").clone(),
            )
        super()._reset_idx(env_ids)
        indices = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        origins = self.scene.env_origins[indices]
        self.scene["base_camera"].set_world_poses_from_view(
            eyes=origins + origins.new_tensor(self._camera_eye),
            targets=origins + origins.new_tensor(self._camera_target),
            env_ids=indices,
        )

    @override
    def step(
        self, action: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._final, self._flags = None, None
        self._stepping = True
        try:
            obs, reward, terminated, truncated, extras = super().step(action)
        finally:
            self._stepping = False
        if self._flags is None:
            success = self.termination_manager.get_term("success").clone()
        else:
            terminated, truncated, success = self._flags
        return (
            obs,
            reward,
            terminated,
            truncated,
            {
                **extras,
                "final_observation": self._final,
                "success": success,
            },
        )


def make_env(config: ArenaLiberoConfig) -> ArenaLiberoEnv:
    """Compose a registered task (or the original bowl smoke) with native Arena APIs.

    Parameters
    ----------
    config : ArenaLiberoConfig
        Local asset paths and batched rollout settings. AppLauncher must already
        be running with cameras enabled; its lifetime belongs to the caller.

    Returns
    -------
    ArenaLiberoEnv
        GPU transitions with float32 state [N,8], task-sized critic [N,D], and two uint8
        RGB images [N,H,W,3]. Actions are normalized relative TCP commands [N,7].
    """
    spec = get_task(config.task_id) if config.task_id is not None else None
    description = IsaacLabArenaEnvironment(
        name="Carrot-BowlOnPlate-v0" if spec is None else f"Carrot-{spec.name}-v0",
        scene=KitchenScene(config) if spec is None else TaskScene(config, spec),
        embodiment=Panda(config, spec),
        task=BowlOnPlateTask(config.max_episode_steps)
        if spec is None
        else LiberoTask(spec, config.max_episode_steps),
    )
    builder = ArenaEnvBuilder(
        description,
        ArenaEnvBuilderCfg(
            num_envs=config.num_envs,
            env_spacing=8.0,
            seed=config.seed,
            device=config.device,
            solve_relations=False,
            presets="physx",
        ),
    )
    _, cfg, kwargs = builder.build_registered()
    cfg.sim.device = config.device
    cfg.sim.dt = 0.02 / config.physics_substeps
    cfg.decimation = config.physics_substeps
    cfg.sim.render_interval = config.physics_substeps
    cfg.num_rerenders_on_reset = 4
    cfg.is_finite_horizon = False
    if spec is not None:
        kwargs.update(camera_eye=spec.camera_eye, camera_target=spec.camera_target)
    env = ArenaLiberoEnv(
        TransitionEnv(cfg, **kwargs),
        image_size=config.image_size,
        critic_size=52 if spec is None else spec.critic_size,
        task_name="put_the_black_bowl_on_the_plate" if spec is None else spec.task_id,
        task_description="put the black bowl on the plate" if spec is None else spec.language,
    )
    try:
        # Initialize articulated rendering before exposing the first reset observation.
        env.reset(seed=config.seed)
        neutral = torch.zeros(config.num_envs, 7, device=env.device)
        neutral[:, 6] = -1
        for _ in range(5):
            env.step(neutral)
        env.reset(seed=config.seed)
    except BaseException:
        env.close()
        raise
    return env
