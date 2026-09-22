from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import torch

Observation = dict[str, torch.Tensor | dict[str, torch.Tensor]]


@dataclass(frozen=True)
class IsaacLiberoConfig:
    """Lightwheel bowl-on-plate task configuration, independent of the Isaac runtime.

    Parameters
    ----------
    num_envs : int
        Environments in one simulator process; one renderer owns the selected GPU.
    seed : int
        Upstream scene/placement seed at construction, not a reset-time reseed.
    max_episode_steps : int
        Time limit in policy steps, with timeout bootstrapping enabled.
    translation_scale : float
        Metres per normalized action component, in the robot base frame.
    rotation_scale : float
        Radians per normalized rotation-vector component, in the robot base frame.
    """

    num_envs: int = 4
    device: str = "cuda:0"
    seed: int = 42
    image_size: int = 256
    max_episode_steps: int = 400
    translation_scale: float = 0.02
    rotation_scale: float = 0.1

    def __post_init__(self) -> None:
        for name, value in (
            ("num_envs", self.num_envs),
            ("image_size", self.image_size),
            ("max_episode_steps", self.max_episode_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for value in (self.translation_scale, self.rotation_scale):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("action scales must be finite and positive")


class LiberoBackend(Protocol):
    num_envs: int
    device: str
    single_observation_space: gym.Space
    observation_space: gym.Space

    def reset(self, *, env_ids: torch.Tensor) -> tuple[Observation, dict[str, Any]]: ...

    def step(
        self, action: torch.Tensor
    ) -> tuple[Observation, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]: ...

    def close(self) -> None: ...


def clone_observation(observation: Observation) -> Observation:
    """Own all tensor storage in a two-level Isaac observation dictionary.

    Parameters
    ----------
    observation : dict
        Batched tensors, optionally grouped one level deep.

    Returns
    -------
    dict
        Detached snapshots safe to retain across steps and resets.
    """
    return {
        key: {name: tensor.detach().clone() for name, tensor in value.items()}
        if isinstance(value, dict)
        else value.detach().clone()
        for key, value in observation.items()
    }


class LiberoVectorEnv:
    """GPU tensor environment with same-step autoreset and sparse success reward.

    Parameters
    ----------
    backend : LiberoBackend
        Isaac backend which captures ``final_observation`` before autoreset and
        supplies a boolean ``success`` termination-term tensor.

    Notes
    -----
    Actions are float32 ``[N, 7]`` in [-1, 1]: base-frame translation/rotvec
    increments and gripper (negative=open, nonnegative=close). Isaac applies the
    configured scales once, then differential IK; this is not MuJoCo OSC parity.
    Returned observations belong to the next episode for done slots. Read
    ``info['final_observation']`` under ``info['_final_observation']`` for the
    transition's actual next state. Bootstrap with ``~terminated`` and stop GAE
    recursion at ``terminated | truncated``. All returned tensors own storage.
    """

    task_name = "L90K1PutTheBlackBowlOnThePlate"
    task_description = "put the black bowl on the plate."
    metadata = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}

    def __init__(self, backend: LiberoBackend) -> None:
        self.backend = backend
        self.num_envs = backend.num_envs
        self.device = torch.device(backend.device)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = backend.single_observation_space
        self.observation_space = backend.observation_space
        self._returns = torch.zeros(self.num_envs, device=self.device)
        self._lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ready = False
        self._closed = False

    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[Observation, dict[str, Any]]:
        """Reset selected slots without advancing physics in the other slots.

        Parameters
        ----------
        env_ids : torch.Tensor | None
            Unique int64 indices, or None for all slots. First reset must be full.

        Returns
        -------
        tuple
            Full-batch observation snapshot and task prompts.
        """
        self._check_open()
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.dtype != torch.long or env_ids.ndim != 1 or env_ids.numel() == 0:
            raise ValueError("env_ids must be a nonempty 1D int64 tensor")
        env_ids = env_ids.to(self.device)
        if (
            (env_ids < 0).any()
            or (env_ids >= self.num_envs).any()
            or env_ids.unique().numel() != env_ids.numel()
        ):
            raise ValueError("env_ids must be unique and within the batch")
        if not self._ready and env_ids.numel() != self.num_envs:
            raise RuntimeError("the first reset must include every environment")
        observation, _ = self.backend.reset(env_ids=env_ids)
        self._returns[env_ids] = 0
        self._lengths[env_ids] = 0
        self._ready = True
        return clone_observation(observation), {"prompt": (self.task_description,) * self.num_envs}

    def step(
        self, action: torch.Tensor
    ) -> tuple[Observation, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Advance every slot by one control step and automatically reset done slots.

        Parameters
        ----------
        action : torch.Tensor
            Finite normalized float32 tensor with shape ``[N, 7]``.

        Returns
        -------
        tuple
            Observation, sparse reward ``[N]``, terminated ``[N]``, truncated
            ``[N]``, and info. Info contains final observations, success and episode
            return/length; masks identify valid final-observation and episode rows.
        """
        self._check_open()
        if not self._ready:
            raise RuntimeError("reset the environment before stepping")
        if action.shape != (self.num_envs, 7) or action.dtype != torch.float32:
            raise ValueError("action must be float32 with shape [num_envs, 7]")
        if not torch.isfinite(action).all() or (action.abs() > 1).any():
            raise ValueError("action must be finite and within [-1, 1]; clip in the policy")
        native_action = action.detach().to(self.device).clone()
        # Isaac uses the opposite gripper sign; zero follows our close convention.
        native_action[:, 6] = torch.where(native_action[:, 6] < 0, 1.0, -1.0)
        observation, _, terminated, truncated, backend_info = self.backend.step(native_action)
        terminated = terminated.clone()
        truncated = truncated.clone()
        done = terminated | truncated
        # Isaac's term history can retain a prior success in slots that are still running.
        success = backend_info["success"] & terminated
        reward = success.to(torch.float32)
        final_observation = backend_info["final_observation"]
        if done.any() and final_observation is None:
            raise RuntimeError("backend lost the final observation before autoreset")
        self._returns += reward
        self._lengths += 1
        info = {
            "prompt": (self.task_description,) * self.num_envs,
            "success": success,
            "final_observation": (
                clone_observation(final_observation) if final_observation is not None else None
            ),
            "_final_observation": done.clone(),
            "episode": {"r": self._returns.clone(), "l": self._lengths.clone()},
            "_episode": done.clone(),
            "time_outs": truncated & ~terminated,
        }
        self._returns[done] = 0
        self._lengths[done] = 0
        return clone_observation(observation), reward, terminated, truncated, info

    def close(self) -> None:
        """Release the backend once; the caller still owns the Isaac application."""
        if not self._closed:
            self.backend.close()
            self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("environment is closed")
