from dataclasses import replace
from pathlib import Path
from typing import Any, override

from lerobot.datasets import LeRobotDatasetMetadata
from torch.utils.data import Dataset

from carrot.data.dataset_spec import SFTDatasetSpec
from examples.so101_real.frame_index import render_frame_index
from recipes.pi05_sft_so101_state_jitter.augmentation import build_dataset as build_jitter_dataset


class FrameIndexDataset(Dataset[dict[str, Any]]):
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
        image = render_frame_index(index)
        assert index < self.frames, "frame index outside single episode"
        return self.source[index] | {"observation/wrist_image": image}


def build_dataset(**kwargs: Any) -> SFTDatasetSpec:
    """Replace wrist video with frame index after the existing state jitter wrapper.

    Parameters
    ----------
    kwargs : Any
        Existing state-jitter factory arguments; requires one episode and wrist-only views.

    Returns
    -------
    SFTDatasetSpec
        Original targets, statistics and jittered state with a synthetic wrist image.

    Raises
    ------
    AssertionError
        If metadata is not one episode, its length disagrees, or views are incompatible.
    """
    assert kwargs["base_image_key"] is None and kwargs["wrist_image_key"] is not None, (
        "frame index training requires wrist-only input"
    )
    root = Path(kwargs["root"]) if kwargs["root"] is not None else None
    metadata = LeRobotDatasetMetadata(
        kwargs["repo_id"], root=root, revision=kwargs["revision"],
    )
    assert metadata.total_episodes == 1, "frame index training supports exactly one episode"
    spec = build_jitter_dataset(**kwargs)
    return replace(spec, dataset=FrameIndexDataset(spec.dataset, metadata.total_frames))
