"""RoboTwin Aloha observation and action adapters."""

from typing import Any

import numpy as np
import torch

from carrot.data.robotwin import robotwin_preprocess

CAMERAS = {
    "cam_high": "base_0_rgb",
    "cam_left_wrist": "left_wrist_0_rgb",
    "cam_right_wrist": "right_wrist_0_rgb",
}


class AlohaInputs:
    """Map unbatched RGB CHW images and a raw 14-dimensional RoboTwin state."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("state must be finite with shape (14,)")
        state, _ = robotwin_preprocess(torch.tensor(state, device=self.device), None)
        images = {}
        for source, target in CAMERAS.items():
            image = np.asarray(data["images"][source])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"{source} must be RGB uint8 with shape (3, H, W)")
            if min(image.shape[1:]) < 1:
                raise ValueError(f"{source} must have non-empty spatial dimensions")
            images[target] = image.copy()
        return {
            "state": state,
            "image": images,
            "image_mask": dict.fromkeys(images, True),
            "prompt": data["prompt"],
        }


class AlohaOutputs:
    """Convert absolute PI-space actions back to raw RoboTwin joint commands."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = data["actions"][..., :14].clone()
        actions *= actions.new_tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
        actions[..., [6, 13]] = (actions[..., [6, 13]] + 0.5476 + 0.6213) / (1.4910 + 0.6213)
        return {"actions": actions}
