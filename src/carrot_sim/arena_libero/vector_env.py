from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import torch

Observation = dict[str, torch.Tensor]


class Backend(Protocol):
    num_envs: int
    device: str

    def reset(
        self, *, seed: int | None = None, env_ids: torch.Tensor | None = None
    ) -> tuple[dict, dict]: ...

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]: ...

    def close(self) -> None: ...


def snapshot(observation: Observation) -> Observation:
    return {key: value.clone() for key, value in observation.items()}


class ArenaLiberoEnv:
    """Expose owned GPU observations and same-step autoreset transitions to RL.

    Parameters
    ----------
    backend : Backend
        Isaac runtime whose policy observation contains state, image and wrist_image,
        and whose critic group is a privileged state tensor.
    """

    task_name = "put_the_black_bowl_on_the_plate"

    def __init__(
        self,
        backend: Backend,
        image_size: int = 256,
        critic_size: int = 52,
        task_name: str = "put_the_black_bowl_on_the_plate",
        task_description: str = "put the black bowl on the plate",
    ) -> None:
        self.task_name = task_name
        self.task_description = task_description
        self.backend = backend
        self.num_envs = backend.num_envs
        self.device = torch.device(backend.device)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
                "critic": gym.spaces.Box(-np.inf, np.inf, (critic_size,), dtype=np.float32),
                "image": gym.spaces.Box(0, 255, (image_size, image_size, 3), dtype=np.uint8),
                "wrist_image": gym.spaces.Box(0, 255, (image_size, image_size, 3), dtype=np.uint8),
            }
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

    def reset(self, env_ids: torch.Tensor | None = None, seed: int | None = None) -> Observation:
        if env_ids is not None:
            if env_ids.ndim != 1 or env_ids.dtype != torch.long or env_ids.device != self.device:
                raise ValueError("env_ids must be a one-dimensional int64 tensor on the env device")
            if env_ids.numel() == 0 or torch.any((env_ids < 0) | (env_ids >= self.num_envs)):
                raise ValueError("env_ids must be nonempty and in range")
            if env_ids.unique().numel() != env_ids.numel():
                raise ValueError("env_ids must be unique")
        observation, _ = self.backend.reset(seed=seed, env_ids=env_ids)
        return flatten_observation(observation)

    def step(
        self, action: torch.Tensor
    ) -> tuple[Observation, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Advance all slots; final_observation is valid only where final_mask is true.

        Parameters
        ----------
        action : torch.Tensor
            Float32 [N,7], bounded to [-1,1], on the environment device. Six relative
            TCP commands followed by gripper: negative opens, nonnegative closes.

        Returns
        -------
        tuple
            Owned observations, rewards, terminated, truncated, and transition info.
            Bootstrap on truncated-only transitions, never on true termination.
        """
        if (
            action.shape != (self.num_envs, 7)
            or action.dtype != torch.float32
            or action.device != self.device
        ):
            raise ValueError("action must be float32 [num_envs,7] on the env device")
        if not torch.isfinite(action).all() or torch.any(action.abs() > 1):
            raise ValueError("actions must be finite and in [-1,1]")
        command = action.clone()
        command[:, 6] = torch.where(action[:, 6] < 0, 1.0, -1.0)
        obs, reward, terminated, truncated, extras = self.backend.step(command)
        terminated, truncated = terminated.clone(), truncated.clone()
        mask = terminated | truncated
        final = extras["final_observation"]
        if mask.any() and final is None:
            raise RuntimeError("autoreset transition is missing its final observation")
        info = {
            "success": extras["success"].clone(),
            "final_mask": mask,
            "final_observation": flatten_observation(final) if final is not None else None,
            "bootstrap_mask": ~terminated,
        }
        return flatten_observation(obs), reward.clone(), terminated, truncated, info

    def close(self) -> None:
        self.backend.close()


def flatten_observation(observation: dict) -> Observation:
    return snapshot({**observation["policy"], "critic": observation["critic"]})
