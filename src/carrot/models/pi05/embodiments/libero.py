"""LIBERO Panda transforms shared by training and inference."""

from typing import Any

import numpy as np

from carrot.models.pi05 import transforms


def _parse_image(value: Any, name: str) -> np.ndarray:
    image = np.asarray(value)
    assert image.ndim == 3, f"{name} must be an RGB image"
    if np.issubdtype(image.dtype, np.floating):
        assert np.isfinite(image).all(), f"{name} must be finite"
        image = (255 * image).astype(np.uint8)
    assert image.dtype == np.uint8, f"{name} must be uint8 or floating point"
    assert image.shape[0] == 3 or image.shape[-1] == 3, (
        f"{name} must have shape (3, H, W) or (H, W, 3)"
    )
    if image.shape[0] == 3:
        return image.copy()
    return np.transpose(image, (2, 0, 1)).copy()


class LiberoInputs:
    """Map LIBERO Panda data into the canonical PI0.5 data contract."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        assert state.shape == (8,) and np.isfinite(state).all(), (
            "observation/state must be finite with shape (8,)"
        )
        base_image = _parse_image(data["observation/image"], "observation/image")
        wrist_image = _parse_image(
            data["observation/wrist_image"], "observation/wrist_image"
        )
        assert base_image.shape[1:] == wrist_image.shape[1:], (
            "LIBERO camera images must have matching spatial dimensions"
        )
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
            assert actions.ndim == 2 and actions.shape[-1] == 7 and np.isfinite(actions).all(), (
                "actions must be finite with shape (H, 7)"
            )
            result["actions"] = actions
        return result


class LiberoOutputs:
    """Crop padded model actions to the seven-dimensional LIBERO command."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"actions": data["actions"][..., :7]}


def create_libero_transform_spec(
    tokenizer: Any,
    norm_stats: dict[str, dict[str, Any]],
    *,
    model_action_dim: int,
    default_prompt: str | None = None,
) -> transforms.Pi05TransformSpec:
    """Build the shared LIBERO Panda transform pipeline.

    Parameters
    ----------
    tokenizer : Any
    norm_stats : dict
        Quantile statistics for 8-dimensional ``state`` and 7-dimensional ``actions``.
    model_action_dim : int
    default_prompt : str | None

    Returns
    -------
    transforms.Pi05TransformSpec
    """
    normalize = transforms.Normalize(
        norm_stats,
        use_quantiles=True,
        dimensions={"state": 8, "actions": 7},
    )
    return transforms.Pi05TransformSpec(
        inputs=(
            transforms.InjectDefaultPrompt(default_prompt),
            LiberoInputs(),
            normalize,
            transforms.ResizeImagesPIL(),
            transforms.PadStatesAndActions(model_action_dim),
            transforms.TokenizePrompt(tokenizer, discrete_state_input=False),
        ),
        outputs=(
            transforms.Unnormalize(norm_stats, use_quantiles=True),
            LiberoOutputs(),
        ),
        action_dim=7,
    )
