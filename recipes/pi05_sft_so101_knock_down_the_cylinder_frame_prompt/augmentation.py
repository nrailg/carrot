from dataclasses import replace
from numbers import Integral
from pathlib import Path
from typing import Any, override

from lerobot.datasets import LeRobotDatasetMetadata
from torch.utils.data import Dataset

from carrot.data.dataset_spec import SFTDatasetSpec
from examples.so101_real.frame_prompt import format_frame_prompt
from recipes.pi05_sft_so101_state_jitter.augmentation import build_dataset as build_jitter_dataset


class FramePromptDataset(Dataset[dict[str, Any]]):
    def __init__(self, source: Dataset, frames: int) -> None:
        assert type(frames) is int and 1 <= frames <= 10000, "frames must be in [1, 10000]"
        assert len(source) == frames, "single episode length must match metadata total_frames"
        self.source = source
        self.frames = frames

    @override
    def __len__(self) -> int:
        return self.frames

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        assert isinstance(index, Integral) and not isinstance(index, bool) and (
            0 <= index < self.frames
        ), "frame index outside single episode"
        sample = self.source[index]
        return sample | {"prompt": format_frame_prompt(sample["prompt"], index)}


def build_dataset(**kwargs: Any) -> SFTDatasetSpec:
    """Append the frame condition to the task after the existing state jitter wrapper.

    Parameters
    ----------
    kwargs : Any
        Existing state-jitter factory arguments; requires one episode and wrist-only views.

    Returns
    -------
    SFTDatasetSpec
        Original images, targets, statistics and jittered state with a conditioned task prompt.

    Raises
    ------
    AssertionError
        If metadata is not one episode, its length disagrees, or views are incompatible.
    """
    assert kwargs["base_image_key"] is None and kwargs["wrist_image_key"] is not None, (
        "frame prompt training requires wrist-only loader input"
    )
    root = Path(kwargs["root"]) if kwargs["root"] is not None else None
    metadata = LeRobotDatasetMetadata(
        kwargs["repo_id"], root=root, revision=kwargs["revision"],
    )
    assert metadata.total_episodes == 1, "frame prompt training supports exactly one episode"
    spec = build_jitter_dataset(**kwargs)
    return replace(spec, dataset=FramePromptDataset(spec.dataset, metadata.total_frames))
