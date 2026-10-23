import json
from pathlib import Path

import numpy as np


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
