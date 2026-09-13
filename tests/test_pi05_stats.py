import numpy as np
import pytest

from carrot.cli.compute_pi05_stats import Statistics, _update_episode


def test_stats_use_future_action_windows_relative_to_current_state() -> None:
    state = np.zeros((2, 14), dtype=np.float32)
    action = np.zeros((2, 14), dtype=np.float32)
    state[:, 0] = [1, 2]
    action[:, 0] = [10, 20]
    state_stats = Statistics(14, sample_stride=1)
    action_stats = Statistics(14, sample_stride=1)

    _update_episode(
        state,
        action,
        action_horizon=3,
        state_stats=state_stats,
        action_stats=action_stats,
    )

    assert state_stats.count == 2
    assert action_stats.count == 6
    assert action_stats.result()["mean"][0] == pytest.approx((9 + 19 + 19 + 18 + 18 + 18) / 6)
