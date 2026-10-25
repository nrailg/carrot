import hashlib
from pathlib import Path

import numpy as np


def file_sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def paired_noise(key: str, chunk: int, action_dim: int) -> np.ndarray:
    """Return model-independent noise for the same episode and inference call.

    Parameters
    ----------
    key : str
        Stable suite/task/episode identity, excluding the checkpoint name.
    chunk, action_dim : int

    Returns
    -------
    numpy.ndarray
        Float32 noise of shape (10, action_dim).
    """
    assert chunk >= 0 and action_dim >= 7
    seed = int.from_bytes(hashlib.sha256(f'{key}/{chunk}'.encode()).digest()[:8], 'little')
    return np.random.Generator(np.random.PCG64(seed)).standard_normal((10, action_dim)).astype(np.float32)
