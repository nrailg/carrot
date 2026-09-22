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
from carrot_sim.arena_libero.vector_env import ArenaLiberoEnv


class TransitionEnv(IsaacLabArenaManagerBasedRLEnv):
    def __init__(self, cfg: IsaacLabArenaManagerBasedRLEnvCfg, **kwargs: Any) -> None:
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
            eyes=origins + origins.new_tensor([2.48, -2.65, 1.65]),
            targets=origins + origins.new_tensor([2.48, -2.04, 0.80]),
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
    """Compose a kitchen, Panda and bowl-on-plate task with native Arena APIs.

    Parameters
    ----------
    config : ArenaLiberoConfig
        Local asset paths and batched rollout settings. AppLauncher must already
        be running with cameras enabled; its lifetime belongs to the caller.

    Returns
    -------
    ArenaLiberoEnv
        GPU transitions with float32 state [N,8], critic [N,52], and two uint8
        RGB images [N,H,W,3]. Actions are normalized relative TCP commands [N,7].
    """
    description = IsaacLabArenaEnvironment(
        name="Carrot-BowlOnPlate-v0",
        scene=KitchenScene(config),
        embodiment=Panda(config),
        task=BowlOnPlateTask(config.max_episode_steps),
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
    cfg.sim.dt = 0.01
    cfg.decimation = 2
    cfg.sim.render_interval = 2
    cfg.num_rerenders_on_reset = 4
    cfg.is_finite_horizon = False
    env = ArenaLiberoEnv(TransitionEnv(cfg, **kwargs), image_size=config.image_size)
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
