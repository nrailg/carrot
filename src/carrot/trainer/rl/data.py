import json
import shutil
from pathlib import Path

import numpy as np


def reuse_demonstrations(source: str, destination: Path, limit: int) -> dict:
    """Copy committed successful episodes from an interrupted collection into a fresh run.

    Parameters
    ----------
    source : str
        Prior renderer's demos directory. A result.json written last commits an episode;
        an unfinished directory without that marker is explicitly excluded.
    destination : pathlib.Path
        Existing empty directory; the source is never modified.
    limit : int
        Maximum accepted successes. Corrupt committed episodes are errors.

    Returns
    -------
    dict
        Provenance, completed episode metadata and a fresh training seed range.
        Only reused successes count against this run's attempt budget.
    """
    root = Path(source)
    assert root.is_dir() and not list(destination.iterdir())
    episodes = []
    for path in sorted(root.glob("episode-*")):
        marker = path / "result.json"
        if not marker.is_file():
            continue
        info = json.loads(marker.read_text())
        assert info["success"], f"unsuccessful committed episode: {path}"
        assert len(episodes) < limit, f"source exceeds requested {limit} demonstrations"
        shutil.copytree(path, destination / f"episode-{len(episodes):04d}")
        episodes.append(info)
    assert episodes, f"no committed successes to reuse: {source}"
    Demonstrations(str(destination))
    # 老进程可能还有未落盘的并行轨迹，为新采集预留不重叠的种子区间。
    next_seed = max(info["seed"] for info in episodes) + 1000
    assert next_seed < 1_000_000
    return {
        "source": str(root.resolve()),
        "saved": len(episodes),
        "episodes": episodes,
        "next_seed": next_seed,
    }


class Demonstrations:
    """Memory-map successful episodes and mask action targets at subtask boundaries.

    Parameters
    ----------
    directory : str
        Each episode contains pre-action observations, actions, phase IDs and success metadata.
    horizon : int
    """

    def __init__(self, directory: str, horizon: int = 10) -> None:
        self.horizon = horizon
        self.episodes = []
        self.index = []
        for episode in sorted(Path(directory).glob("episode-*")):
            info = json.loads((episode / "result.json").read_text())
            assert info["success"], f"unsuccessful demonstration: {episode}"
            arrays = {
                p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in episode.glob("*.npy")
            }
            required = {"actions", "phase", "state", "prompt", "image", "wrist_image"}
            assert required <= arrays.keys(), f"missing demonstration arrays: {episode}"
            count = len(arrays["actions"])
            assert count > 0 and arrays["actions"].shape == (count, 7)
            assert all(len(value) == count for value in arrays.values())
            assert np.isfinite(arrays["actions"]).all()
            self.index.extend((len(self.episodes), i) for i in range(count))
            self.episodes.append(arrays)
        assert self.index, f"no successful demonstrations: {directory}"

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[dict, int]:
        episode, offset = self.index[index]
        data = self.episodes[episode]
        length = 1
        while (
            length < self.horizon
            and offset + length < len(data["phase"])
            and data["phase"][offset + length] == data["phase"][offset]
        ):
            length += 1
        action = np.zeros((self.horizon, 7), dtype=np.float32)
        action[:length] = data["actions"][offset : offset + length]
        return {
            "observation/image": data["image"][offset].copy(),
            "observation/wrist_image": data["wrist_image"][offset].copy(),
            "observation/state": data["state"][offset].copy(),
            "prompt": str(data["prompt"][offset]),
            "actions": action,
        }, length
