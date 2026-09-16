"""RoboTwin joint and gripper conversions shared by dataset and policy adapters."""

import numpy as np
import torch


def robotwin_preprocess(
    state: torch.Tensor | np.ndarray, actions: torch.Tensor | np.ndarray | None
) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
    """Convert RobotWin Aloha labels into PI0.5 SFT targets.

    Parameters
    ----------
    state : torch.Tensor | numpy.ndarray
    actions : torch.Tensor | numpy.ndarray | None
        Pass None for observation-only inference.

    Returns
    -------
    tuple[torch.Tensor | numpy.ndarray, torch.Tensor | numpy.ndarray | None]
    """
    if isinstance(state, torch.Tensor):
        flip = state.new_tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
        state = state.clone()
        state[..., :14] *= flip
        linear = 0.01844 + state[..., [6, 13]] * (0.05800 - 0.01844)
        ratio = (0.022**2 + linear**2 - 0.036**2) / (2 * 0.022 * linear)
        state[..., [6, 13]] = (torch.asin(torch.clamp(ratio, -1, 1)) - 0.5476) / (1.6296 - 0.5476)
    else:
        flip = np.asarray([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=np.float32)
        state = state.copy()
        state[..., :14] *= flip
        linear = 0.01844 + state[..., [6, 13]] * (0.05800 - 0.01844)
        ratio = (0.022**2 + linear**2 - 0.036**2) / (2 * 0.022 * linear)
        state[..., [6, 13]] = (np.arcsin(np.clip(ratio, -1, 1)) - 0.5476) / (1.6296 - 0.5476)
    if actions is None:
        return state, None
    actions = actions.clone() if isinstance(actions, torch.Tensor) else actions.copy()
    actions[..., :14] *= flip
    actions[..., [6, 13]] = (-0.6213 + actions[..., [6, 13]] * (1.4910 + 0.6213)) - 0.5476
    delta_mask = [True] * 6 + [False] + [True] * 6 + [False]
    actions[..., delta_mask] -= state[..., delta_mask][..., None, :]
    return state, actions
