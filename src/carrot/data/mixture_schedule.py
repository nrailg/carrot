import numpy as np


def valid_lengths(episode_ids: np.ndarray, horizon: int = 10) -> np.ndarray:
    """Count valid future frames without crossing a contiguous episode boundary.

    Parameters
    ----------
    episode_ids : numpy.ndarray
        One episode identifier per frame in dataset order.
    horizon : int

    Returns
    -------
    numpy.ndarray
        Positive valid action counts, shape (frames,).
    """
    assert episode_ids.ndim == 1 and len(episode_ids) and horizon > 0
    ends = np.r_[np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1, len(episode_ids)]
    starts = np.r_[0, ends[:-1]]
    assert len(np.unique(episode_ids[starts])) == len(starts), "noncontiguous episodes"
    return np.minimum(np.repeat(ends, ends - starts) - np.arange(len(episode_ids)), horizon)


def make_schedule(
    libero_lengths: np.ndarray,
    draw_lengths: np.ndarray,
    *,
    mixed: bool,
    seed: int = 1000,
    steps: int = 200,
    global_batch_size: int = 256,
    dp_size: int = 8,
) -> dict[str, np.ndarray]:
    """Construct exact source quotas and global valid-frame loss normalization.

    Parameters
    ----------
    libero_lengths, draw_lengths : numpy.ndarray
        Valid horizon lengths for each source sample.
    mixed : bool
        Equal source counts when true; otherwise LIBERO only.
    seed, steps, global_batch_size, dp_size : int
        DistributedSampler must use shuffle=False and this world size.

    Returns
    -------
    dict[str, numpy.ndarray]
        Flattened domain, source_index, valid_length and loss_weight arrays.
        Loss is the microbatch mean of weighted per-sample valid-frame sums.
    """
    assert steps > 0 and global_batch_size % (2 * dp_size) == 0
    assert all(len(a) and np.all((a >= 1) & (a <= 10)) for a in (libero_lengths, draw_lengths))
    count = steps * global_batch_size
    domain = (np.arange(count) // dp_size) % 2 if mixed else np.zeros(count, np.int64)
    indices = np.empty(count, np.int64)
    lengths = np.empty(count, np.int64)
    for source, source_lengths in enumerate((libero_lengths, draw_lengths)):
        selected = domain == source
        rng = np.random.default_rng(seed + source)
        indices[selected] = rng.integers(len(source_lengths), size=int(selected.sum()))
        lengths[selected] = source_lengths[indices[selected]]
    weights = np.empty(count, np.float64)
    for start in range(0, count, global_batch_size):
        for source in range(2 if mixed else 1):
            selected = domain[start : start + global_batch_size] == source
            denominator = lengths[start : start + global_batch_size][selected].sum()
            weights[start : start + global_batch_size][selected] = (
                global_batch_size * (0.5 if mixed else 1.0) / denominator
            )
    return {
        "domain": domain,
        "source_index": indices,
        "valid_length": lengths,
        "loss_weight": weights.astype(np.float32),
    }
