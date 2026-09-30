"""SO101 transforms shared by PI0.5 training and inference."""

from typing import Any

import numpy as np

from carrot.models.pi05 import transforms


def _parse_image(value: Any, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} must be an RGB image")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or (image < 0).any() or (image > 1).any():
            raise ValueError(f"{name} floating point pixels must be in [0, 1]")
        image = (255 * image).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 or floating point")
    if image.shape[0] == 3:
        return image.copy()
    if image.shape[-1] == 3:
        return np.transpose(image, (2, 0, 1)).copy()
    raise ValueError(f"{name} must have shape (3, H, W) or (H, W, 3)")


class SO101Inputs:
    """Map six joints and available RGB views; omitted views have a false mask."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError("observation/state must be finite with shape (6,)")
        camera_keys = {
            "base_0_rgb": "observation/image", "left_wrist_0_rgb": "observation/wrist_image",
        }
        images = {name: _parse_image(data[key], key)
                  for name, key in camera_keys.items() if key in data}
        assert images, "SO101 requires at least one camera observation"
        empty = np.zeros_like(next(iter(images.values())))
        result = {
            "state": state,
            "image": {
                "base_0_rgb": images.get("base_0_rgb", empty),
                "left_wrist_0_rgb": images.get("left_wrist_0_rgb", empty),
                "right_wrist_0_rgb": empty,
            },
            "image_mask": {
                "base_0_rgb": "base_0_rgb" in images,
                "left_wrist_0_rgb": "left_wrist_0_rgb" in images,
                "right_wrist_0_rgb": False,
            },
            "prompt": data["prompt"],
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 6 or not np.isfinite(actions).all():
                raise ValueError("actions must be finite with shape (H, 6)")
            result["actions"] = actions
        return result


class SO101Outputs:
    """Crop padded model actions to six absolute SO101 joint commands."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"actions": data["actions"][..., :6]}


def create_so101_transform_spec(
    tokenizer: Any,
    norm_stats: dict[str, dict[str, Any]],
    *,
    model_action_dim: int,
    discrete_state_input: bool = True,
    default_prompt: str | None = None,
) -> transforms.Pi05TransformSpec:
    """Build a quantile-normalized SO101 absolute-action transform pipeline.

    Parameters
    ----------
    tokenizer : Any
    norm_stats : dict
        Quantile statistics for six-dimensional ``state`` and ``actions``.
    model_action_dim : int
    discrete_state_input : bool
    default_prompt : str | None

    Returns
    -------
    transforms.Pi05TransformSpec
    """
    if model_action_dim < 6:
        raise ValueError("SO101 requires model_action_dim >= 6")
    normalize = transforms.Normalize(
        norm_stats,
        use_quantiles=True,
        dimensions={"state": 6, "actions": 6},
    )
    return transforms.Pi05TransformSpec(
        inputs=(
            transforms.InjectDefaultPrompt(default_prompt),
            SO101Inputs(),
            normalize,
            transforms.ResizeImagesPIL(),
            transforms.PadStatesAndActions(model_action_dim),
            transforms.TokenizePrompt(tokenizer, discrete_state_input=discrete_state_input),
        ),
        outputs=(
            transforms.Unnormalize(norm_stats, use_quantiles=True),
            SO101Outputs(),
        ),
        action_dim=6,
    )
