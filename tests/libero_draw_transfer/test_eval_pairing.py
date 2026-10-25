import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'recipes/pi05_libero_draw_sft_transfer'))
from eval_common import paired_noise


def test_noise_is_independent_of_call_order() -> None:
    # 相同episode与调用号必须复现，不能因另一模型提前结束而移动随机流。
    # 用逆序与其他任务调用模拟不同模型的执行顺序。
    first = paired_noise('libero_goal/2/7', 3, 32)
    paired_noise('draw/1000000', 900, 32)
    paired_noise('libero_goal/2/7', 2, 32)
    second = paired_noise('libero_goal/2/7', 3, 32)
    # 逐位一致且满足模型noise契约，才能把结果作为配对比较。
    np.testing.assert_array_equal(first, second)
    assert first.shape == (10, 32) and first.dtype == np.float32
    assert np.isfinite(first).all()


def test_calls_and_episodes_use_distinct_noise() -> None:
    # 不同初态和调用号不能重复同一块noise，否则会引入额外相关性。
    # 固定一个任务，只分别改变episode或调用号。
    samples = [paired_noise(key, chunk, 32) for key, chunk in
               [('libero_goal/2/7', 0), ('libero_goal/2/8', 0), ('libero_goal/2/7', 1)]]
    # 比较完整字节，而不是只检查某一维，避免偶然相等造成误判。
    assert len({sample.tobytes() for sample in samples}) == 3
