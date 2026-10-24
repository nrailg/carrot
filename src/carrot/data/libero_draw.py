import json
from pathlib import Path
from typing import Any, override

import numpy as np
from torch.utils.data import Dataset

from carrot.data.dataset_spec import SFTDatasetSpec
from carrot.data.libero import build_dataset as build_libero


class DrawDataset(Dataset):
    """Read successful three-color demonstrations with phase-bounded action targets.

    Parameters
    ----------
    root : str
        Immutable episode directories containing pre-action observations and 7D actions.
    """

    def __init__(self, root: str) -> None:
        self.episodes = []
        self.index = []
        self.lengths = []
        self.seeds = []
        for path in sorted(Path(root).glob("episode-*")):
            info = json.loads((path / "result.json").read_text())
            assert info["success"] and info["picks"] == info["returns"] == 3, path
            self.seeds.append(info["seed"])
            data = {
                name: np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                for name in ("image", "wrist_image", "state", "prompt", "phase", "actions")
            }
            count = len(data["actions"])
            assert data["actions"].shape == (count, 7) and count > 0, path
            assert all(len(a) == count for a in data.values()), path
            assert np.isfinite(data["actions"]).all(), path
            boundaries = np.r_[np.flatnonzero(np.diff(data["phase"]) != 0) + 1, count]
            starts = np.r_[0, boundaries[:-1]]
            lengths = np.minimum(np.repeat(boundaries, boundaries - starts) - np.arange(count), 10)
            self.index.extend((len(self.episodes), frame) for frame in range(count))
            self.lengths.extend(lengths)
            self.episodes.append(data)
        assert self.index and len(self.seeds) == len(set(self.seeds)), root
        self.lengths = np.asarray(self.lengths, np.int64)

    def __len__(self) -> int:
        return len(self.index)

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, frame = self.index[index]
        data = self.episodes[episode]
        length = int(self.lengths[index])
        actions = np.zeros((10, 7), np.float32)
        actions[:length] = data["actions"][frame : frame + length]
        return {
            "observation/image": data["image"][frame].copy(),
            "observation/wrist_image": data["wrist_image"][frame].copy(),
            "observation/state": data["state"][frame].copy(),
            "prompt": str(data["prompt"][frame]),
            "actions": actions,
            "action_is_pad": np.arange(10) >= length,
        }


class ScheduledDataset(Dataset):
    def __init__(self, libero: Dataset, draw: DrawDataset, manifest: str) -> None:
        self.sources = (libero, draw)
        with np.load(manifest, allow_pickle=False) as archive:
            self.schedule = {key: archive[key] for key in archive.files}
        self.count = len(self.schedule["domain"])
        assert all(len(a) == self.count for a in self.schedule.values())

    def __len__(self) -> int:
        return self.count

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        domain = int(self.schedule["domain"][index])
        sample = dict(self.sources[domain][int(self.schedule["source_index"][index])])
        count = int((~np.asarray(sample["action_is_pad"], dtype=bool)).sum())
        assert count == self.schedule["valid_length"][index], (
            f"action boundary differs from manifest at {index}: {count}"
        )
        return {
            **sample,
            "loss_weight": self.schedule["loss_weight"][index],
            "sample_domain": domain,
        }


def build_dataset(*, root: str, draw_root: str, manifest: str) -> SFTDatasetSpec:
    """Bind a frozen sampling schedule to LIBERO and drawing data.

    Parameters
    ----------
    root, draw_root : str
        Local LIBERO and verified drawing dataset directories.
    manifest : str
        NPZ written by this experiment's preparation script; requires shuffle=False.

    Returns
    -------
    SFTDatasetSpec
        Uses the unchanged LIBERO input transforms and normalization contract.
    """
    libero = build_libero(root=root)
    draw = DrawDataset(draw_root)
    return SFTDatasetSpec(
        dataset=ScheduledDataset(libero.dataset, draw, manifest),
        collate_fn=None,
        state_stats=libero.state_stats,
        action_stats=libero.action_stats,
        image_keys=libero.image_keys,
        state_key=libero.state_key,
        action_key=libero.action_key,
        task_key=libero.task_key,
        embodiment="libero",
    )
