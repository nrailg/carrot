"""RoboTwin ALOHA transforms shared by training and inference."""

from typing import Any

import numpy as np
import torch

from carrot.data.robotwin import robotwin_preprocess
from carrot.models.pi05 import transforms

CAMERAS = {
    "cam_high": "base_0_rgb",
    "cam_left_wrist": "left_wrist_0_rgb",
    "cam_right_wrist": "right_wrist_0_rgb",
}


class AlohaInputs:
    """Map RoboTwin ALOHA data into the canonical PI0.5 data contract."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = data["state"]
        actions = data.get("actions")
        if isinstance(state, torch.Tensor):
            state = state.float()
            if state.shape[-1] != 14 or not torch.isfinite(state).all():
                raise ValueError("state must be finite with last dimension 14")
            if actions is not None:
                actions = torch.as_tensor(actions).float()
                if actions.shape[-1] != 14 or not torch.isfinite(actions).all():
                    raise ValueError("actions must be finite with last dimension 14")
        else:
            state = np.asarray(state, dtype=np.float32)
            if state.shape[-1] != 14 or not np.isfinite(state).all():
                raise ValueError("state must be finite with last dimension 14")
            if actions is not None:
                actions = np.asarray(actions, dtype=np.float32)
                if actions.shape[-1] != 14 or not np.isfinite(actions).all():
                    raise ValueError("actions must be finite with last dimension 14")
        state, actions = robotwin_preprocess(state, actions)

        images = {}
        for source, target in CAMERAS.items():
            image = data["images"][source]
            if isinstance(image, torch.Tensor):
                if image.ndim not in (3, 4) or image.shape[-3] != 3:
                    raise ValueError(f"{source} must have shape (3, H, W) or (B, 3, H, W)")
                if not torch.isfinite(image.float()).all():
                    raise ValueError(f"{source} must be finite")
                image = image.clone()
            else:
                image = np.asarray(image)
                if image.ndim not in (3, 4) or image.shape[-3] != 3:
                    raise ValueError(f"{source} must have shape (3, H, W) or (B, 3, H, W)")
                if not np.isfinite(image).all():
                    raise ValueError(f"{source} must be finite")
                image = image.copy()
            if min(image.shape[-2:]) < 1:
                raise ValueError(f"{source} must have non-empty spatial dimensions")
            images[target] = image

        result = {
            "state": state,
            "image": images,
            "image_mask": dict.fromkeys(images, True),
            "prompt": data["prompt"],
        }
        if actions is not None:
            result["actions"] = actions
        return result


class AlohaOutputs:
    """Convert absolute PI-space actions back to raw RoboTwin joint commands."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = data["actions"][..., :14].clone()
        actions *= actions.new_tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
        actions[..., [6, 13]] = (actions[..., [6, 13]] + 0.5476 + 0.6213) / (
            1.4910 + 0.6213
        )
        return {"actions": actions}


def create_aloha_transform_spec(
    tokenizer: Any,
    norm_stats: dict[str, dict[str, Any]],
    *,
    model_action_dim: int,
    default_prompt: str | None = None,
) -> transforms.Pi05TransformSpec:
    """Build the shared RoboTwin ALOHA transform pipeline.

    Parameters
    ----------
    tokenizer : Any
    norm_stats : dict
        Quantile statistics for 14-dimensional ``state`` and ``actions``.
    model_action_dim : int
    default_prompt : str | None

    Returns
    -------
    transforms.Pi05TransformSpec
    """
    normalize = transforms.Normalize(
        norm_stats,
        use_quantiles=True,
        dimensions={"state": 14, "actions": 14},
    )
    return transforms.Pi05TransformSpec(
        inputs=(
            transforms.InjectDefaultPrompt(default_prompt),
            AlohaInputs(),
            normalize,
            transforms.ResizeImages(),
            transforms.PadStatesAndActions(model_action_dim),
            transforms.TokenizePrompt(tokenizer, discrete_state_input=True),
        ),
        outputs=(
            transforms.Unnormalize(norm_stats, use_quantiles=True),
            transforms.AbsoluteActions(),
            AlohaOutputs(),
        ),
        action_dim=14,
    )
