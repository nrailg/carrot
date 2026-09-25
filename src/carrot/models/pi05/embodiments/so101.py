"""SO101 transforms shared by PI0.5 training and inference."""

from typing import Any

import numpy as np

from carrot.models.pi05 import transforms


def _parse_image(value: Any, name: str) -> np.ndarray:
    image = np.asarray(value)
    assert image.ndim == 3, f"{name} must be an RGB image"
    if np.issubdtype(image.dtype, np.floating):
        assert np.isfinite(image).all() and not (image < 0).any() and not (image > 1).any(), (
            f"{name} floating point pixels must be in [0, 1]"
        )
        image = (255 * image).astype(np.uint8)
    assert image.dtype == np.uint8, f"{name} must be uint8 or floating point"
    assert image.shape[0] == 3 or image.shape[-1] == 3, (
        f"{name} must have shape (3, H, W) or (H, W, 3)"
    )
    if image.shape[0] == 3:
        return image.copy()
    return np.transpose(image, (2, 0, 1)).copy()


class SO101Inputs:
    """Map six joints and two RGB cameras to the PI0.5 model fields."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        assert state.shape == (6,) and np.isfinite(state).all(), (
            "observation/state must be finite with shape (6,)"
        )
        base_image = _parse_image(data["observation/image"], "observation/image")
        wrist_image = _parse_image(data["observation/wrist_image"], "observation/wrist_image")
        result = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": False,
            },
            "prompt": data["prompt"],
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            assert actions.ndim == 2 and actions.shape[-1] == 6 and np.isfinite(actions).all(), (
                "actions must be finite with shape (H, 6)"
            )
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
    assert model_action_dim >= 6, "SO101 requires model_action_dim >= 6"
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
