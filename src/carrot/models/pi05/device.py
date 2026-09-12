"""Small device helpers kept independent of the training backend."""

from __future__ import annotations

import torch


def get_device_name() -> str:
    """Return the autocast device type used by the native PI0.5 model."""
    return "cuda" if torch.cuda.is_available() else "cpu"
