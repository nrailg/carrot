"""Generic transform-driven PI0.5 inference."""

import time
from typing import Any

import numpy as np
import torch

from carrot.models.pi05 import transforms
from carrot.models.pi05.model import PI0Observation, PI0Pytorch


class Pi05Policy:
    """Run PI0.5 sampling through an injected embodiment transform spec.

    Parameters
    ----------
    model : PI0Pytorch
    transform_spec : transforms.Pi05TransformSpec
    device : str
    num_steps : int
        Denoising iterations, independent of the predicted action horizon.
    """

    def __init__(
        self,
        model: PI0Pytorch,
        transform_spec: transforms.Pi05TransformSpec,
        *,
        device: str,
        num_steps: int = 10,
    ) -> None:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if model.config.action_dim < transform_spec.action_dim or model.config.action_horizon < 1:
            raise ValueError("model action contract is incompatible with the transform spec")
        self._model = model.to(device).eval()
        self._device = torch.device(device)
        self._num_steps = num_steps
        self._transform_spec = transform_spec
        self._input_transform = transforms.compose(transform_spec.inputs)
        self._output_transform = transforms.compose(transform_spec.outputs)

    @torch.no_grad()
    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        """Predict one action chunk without modifying the caller's observation.

        Parameters
        ----------
        obs : dict
            Raw observation accepted by the injected embodiment transforms.
        noise : numpy.ndarray | None
            Optional finite noise with shape ``(action_horizon, model_action_dim)``.

        Returns
        -------
        dict
            Environment actions and model inference timing.
        """
        inputs = self._input_transform(dict(obs))
        observation = self._to_observation(inputs)
        sample_noise = None
        if noise is not None:
            noise = np.asarray(noise)
            expected = (self._model.config.action_horizon, self._model.config.action_dim)
            if noise.shape != expected or not np.isfinite(noise).all():
                raise ValueError(f"noise must be finite with shape {expected}")
            sample_noise = torch.as_tensor(noise.copy(), device=self._device, dtype=torch.float32)[
                None
            ]

        start = time.perf_counter()
        actions = self._model.sample_actions(
            self._device, observation, noise=sample_noise, num_steps=self._num_steps
        )
        model_time = time.perf_counter() - start
        expected = (1, self._model.config.action_horizon, self._model.config.action_dim)
        if tuple(actions.shape) != expected or not torch.isfinite(actions).all():
            raise ValueError(f"model actions must be finite with shape {expected}")

        outputs = self._output_transform({"state": observation.state, "actions": actions})
        decoded = outputs["actions"]
        if isinstance(decoded, torch.Tensor):
            decoded = decoded[0].float().cpu().numpy()
        else:
            decoded = np.asarray(decoded[0], dtype=np.float32)
        expected_decoded = (self._model.config.action_horizon, self._transform_spec.action_dim)
        if decoded.shape != expected_decoded or not np.isfinite(decoded).all():
            raise ValueError(f"decoded actions must be finite with shape {expected_decoded}")
        return {
            "actions": decoded,
            "policy_timing": {"infer_ms": model_time * 1000},
        }

    def _to_observation(self, inputs: dict[str, Any]) -> PI0Observation:
        dtype = self._model.action_in_proj.weight.dtype
        return transforms.to_observation(inputs, device=self._device, dtype=dtype)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "action_horizon": self._model.config.action_horizon,
            "action_dim": self._transform_spec.action_dim,
            "num_steps": self._num_steps,
        }
