from dataclasses import replace
from typing import Any

from torch.utils.data import Subset

from carrot.data.dataset_spec import SFTDatasetSpec
from carrot.data.so101 import build_dataset as build_so101_dataset


def build_dataset(*, frame_index: int, repeat_count: int, **kwargs: Any) -> SFTDatasetSpec:
    """Repeat one source observation without changing its action window or stats.

    Parameters
    ----------
    frame_index : int
        Global index into the original SO101 dataset.
    repeat_count : int
        Virtual dataset length; enough samples must remain for distributed batches.
    kwargs : Any
        Arguments passed to the standard SO101 dataset factory.

    Returns
    -------
    SFTDatasetSpec
        Repeated source sample with the original metadata and normalization stats.
    """
    spec = build_so101_dataset(**kwargs)
    assert 0 <= frame_index < len(spec.dataset), "frame_index outside the source dataset"
    assert repeat_count >= 1, "repeat_count must be positive"
    return replace(spec, dataset=Subset(spec.dataset, [frame_index] * repeat_count))
