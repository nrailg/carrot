"""Legacy PI0.5 preprocessing helpers.

TODO: Migrate ``Pi05SFTLossFn`` to the shared transform pipeline, move prompt
tokenization into an independent transform/helper, and remove
``Pi05Preprocessor`` together with its duplicated normalization, padding, and
image preprocessing.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _stats_tensor(stats: dict[str, Any], name: str, device: torch.device) -> torch.Tensor:
    value = stats[name]
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


class Pi05Preprocessor:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @staticmethod
    def _normalize(x: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
        lower = "q01" if "q01" in stats else "min"
        upper = "q99" if "q99" in stats else "max"
        q01 = _stats_tensor(stats, lower, x.device)
        q99 = _stats_tensor(stats, upper, x.device)
        width = min(x.shape[-1], q01.shape[-1])
        head = 2 * (x[..., :width].float() - q01[:width]) / (q99[:width] - q01[:width] + 1e-6) - 1
        return torch.cat((head, x[..., width:].float()), dim=-1)

    @staticmethod
    def _pad_last(x: torch.Tensor, width: int) -> torch.Tensor:
        return F.pad(x, (0, width - x.shape[-1])) if x.shape[-1] < width else x[..., :width]

    @staticmethod
    def _prepare_image(image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        image = image.float()
        if image.max() > 1:
            image = image / 255
        if image.shape[-2:] != (224, 224):
            height, width = image.shape[-2:]
            ratio = max(width / 224, height / 224)
            resized_height = int(height / ratio)
            resized_width = int(width / ratio)
            image = F.interpolate(
                image,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            )
            pad_height = 224 - resized_height
            pad_width = 224 - resized_width
            image = F.pad(
                image,
                (
                    pad_width // 2,
                    pad_width - pad_width // 2,
                    pad_height // 2,
                    pad_height - pad_height // 2,
                ),
            )
        return (2 * image - 1).to(dtype)

    def _tokenize(
        self,
        tasks: Any,
        state: torch.Tensor,
        *,
        discrete_state_input: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(tasks, str):
            tasks = [tasks]
        elif isinstance(tasks, (np.ndarray, torch.Tensor)):
            tasks = tasks.tolist()
        prompts = []
        if len(tasks) != state.shape[0]:
            raise ValueError("prompt batch size must match state batch size")
        discrete = None
        if discrete_state_input:
            bins = torch.linspace(-1, 1, 257, device=state.device)[:-1]
            discrete = torch.bucketize(state, bins) - 1
        for index, task in enumerate(tasks):
            clean_task = str(task).strip().replace("_", " ").replace("\n", " ")
            if discrete is None:
                prompts.append(f"{clean_task}\n")
            else:
                state_text = " ".join(str(int(value)) for value in discrete[index])
                prompts.append(f"Task: {clean_task}, State: {state_text};\nAction: ")
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=200,
            return_tensors="pt",
        )
        return (
            tokens["input_ids"].to(device=state.device, dtype=torch.long),
            tokens["attention_mask"].to(device=state.device, dtype=torch.bool),
        )
