from dataclasses import replace
from typing import Any, override

import numpy as np
from torch.utils.data import Dataset

from carrot.data.dataset_spec import SFTDatasetSpec
from carrot.data.so101 import build_dataset as build_so101_dataset


class StateJitterDataset(Dataset[dict[str, Any]]):
    def __init__(self, source: Dataset, degrees: float) -> None:
        assert np.isfinite(degrees) and degrees > 0, "state jitter must be positive degrees"
        self.source = source
        self.degrees = degrees

    @override
    def __len__(self) -> int:
        return len(self.source)

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.source[index]
        state = np.asarray(sample["observation/state"], dtype=np.float32).copy()
        assert state.shape == (6,) and np.isfinite(state).all()
        # DataLoader seeds NumPy per worker; perturb raw degrees before state tokenization.
        state[:5] += np.random.uniform(-self.degrees, self.degrees, size=5).astype(np.float32)
        return sample | {"observation/state": state}


def build_dataset(*, state_jitter_degrees: float, **kwargs: Any) -> SFTDatasetSpec:
    """Perturb only training state, preserving original targets and normalization stats.

    Parameters
    ----------
    state_jitter_degrees : float
        Each of five angular joints receives independent uniform signed noise in degrees.
    kwargs : Any
        Forwarded to the SO101 dataset factory.

    Returns
    -------
    SFTDatasetSpec
        Raw-state augmentation precedes shared normalization and tokenization.
    """
    spec = build_so101_dataset(**kwargs)
    return replace(spec, dataset=StateJitterDataset(spec.dataset, state_jitter_degrees))
