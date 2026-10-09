from typing import override

import torch
from torch import Tensor

from configuration_pi05_control import PI05ControlConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


class PI05ControlPolicy(PI05Policy):
    config_class = PI05ControlConfig
    name = "pi05_control"

    @override
    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        images, masks = super()._preprocess_images(batch)
        return [torch.zeros_like(image) for image in images], [torch.zeros_like(mask) for mask in masks]
