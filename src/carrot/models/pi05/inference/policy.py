"""Single-observation PI0.5 inference with RoboTwin action decoding."""

import time
from typing import Any

import numpy as np
import torch

from carrot.models.pi05.model import PI0Observation, PI0Policy
from carrot.models.pi05.preprocessing import Pi05Preprocessor

from . import transforms
from .aloha_policy import AlohaInputs, AlohaOutputs


class Pi05Policy(Pi05Preprocessor):
    """Run a RoboTwin policy using checkpoint-matched preprocessing.

    Parameters
    ----------
    model : PI0Policy
        Native model with an action dimension of at least 14.
    tokenizer : Any
        The tokenizer saved by the SFT run.
    norm_stats : dict
        Quantiles keyed by state and actions, each with 14 entries.
    device : str
    num_steps : int
        Denoising iterations, independent of the predicted action horizon.
    default_prompt : str | None
    """

    def __init__(
        self,
        model: PI0Policy,
        tokenizer: Any,
        norm_stats: dict[str, dict[str, Any]],
        *,
        device: str,
        num_steps: int = 10,
        default_prompt: str | None = None,
    ) -> None:
        super().__init__(tokenizer)
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if model.config.action_dim < 14 or model.config.action_horizon < 1:
            raise ValueError("RoboTwin requires action_dim >= 14 and action_horizon >= 1")
        for key in ("state", "actions"):
            lower = np.asarray(norm_stats[key]["q01"], dtype=np.float32)
            upper = np.asarray(norm_stats[key]["q99"], dtype=np.float32)
            if (
                lower.shape != (14,)
                or upper.shape != (14,)
                or not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
                or (upper < lower).any()
            ):
                raise ValueError(f"{key} quantiles must be finite ordered arrays of shape (14,)")
        self._model = model.to(device).eval()
        self._device = torch.device(device)
        self._num_steps = num_steps
        self._default_prompt = default_prompt
        self._input_transform = AlohaInputs(self._device)
        self._normalize_transform = transforms.Normalize(norm_stats)
        self._output_transform = transforms.compose(
            [transforms.Unnormalize(norm_stats), transforms.AbsoluteActions(), AlohaOutputs()]
        )

    @torch.no_grad()
    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        """Predict a full action chunk without modifying the caller's observation.

        Parameters
        ----------
        obs : dict
            Raw state (14,), three RGB uint8 CHW images and a string prompt.
        noise : numpy.ndarray | None
            Optional finite float noise of shape (action_horizon, action_dim).

        Returns
        -------
        dict
            actions: float32 NumPy array (action_horizon, 14);
            policy_timing: elapsed inference milliseconds.
        """
        start = time.perf_counter()
        prompt = obs.get("prompt", self._default_prompt)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("a non-empty prompt or default_prompt is required")
        inputs = self._input_transform({**obs, "prompt": prompt})
        dtype = self._model.action_in_proj.weight.dtype
        inputs["state"] = torch.as_tensor(inputs["state"], device=self._device)[None]
        inputs = self._normalize_transform(inputs)
        state = self._pad_last(inputs["state"], self._model.config.action_dim)
        tokens, token_mask = self._tokenize(prompt, state)
        images = {
            key: self._prepare_image(
                torch.as_tensor(value, device=self._device)[None].float() / 255, dtype
            )
            for key, value in inputs["image"].items()
        }
        observation = PI0Observation(
            images=images,
            image_masks={
                key: torch.ones(1, dtype=torch.bool, device=self._device) for key in images
            },
            state=state.to(dtype),
            tokenized_prompt=tokens,
            tokenized_prompt_mask=token_mask,
        )
        sample_noise = None
        if noise is not None:
            noise = np.asarray(noise)
            expected = (self._model.config.action_horizon, self._model.config.action_dim)
            if noise.shape != expected or not np.isfinite(noise).all():
                raise ValueError(f"noise must be finite with shape {expected}")
            sample_noise = torch.as_tensor(noise.copy(), device=self._device, dtype=torch.float32)[
                None
            ]
        actions = self._model.sample_actions(
            self._device, observation, noise=sample_noise, num_steps=self._num_steps
        )
        expected = (1, self._model.config.action_horizon, self._model.config.action_dim)
        if tuple(actions.shape) != expected or not torch.isfinite(actions).all():
            raise ValueError(f"model actions must be finite with shape {expected}")
        outputs = self._output_transform({"state": state, "actions": actions})
        actions = outputs["actions"][0].float().cpu().numpy()
        if not np.isfinite(actions).all():
            raise ValueError("decoded actions must be finite")
        return {
            "actions": actions,
            "policy_timing": {"infer_ms": (time.perf_counter() - start) * 1000},
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "action_horizon": self._model.config.action_horizon,
            "action_dim": 14,
            "num_steps": self._num_steps,
        }
