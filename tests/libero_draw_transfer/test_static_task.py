import itertools

import numpy as np
import pytest
from multicolor_logic import PaintProgress, generate_task


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_completion_accepts_every_color_order(order) -> None:
    # 同一张三色图的六种绘画顺序都应有效，成功判定不能依赖teacher顺序。
    # 未使用的笔在笔架中，但没有被拿起和画过，不能提前算归还完成。
    progress = PaintProgress(generate_task(17, 3))
    empty = np.zeros(3, dtype=bool)
    reward = 0.0
    assert progress.advance(empty, empty, np.ones(3, dtype=bool)) == 0
    for count, color in enumerate(order, start=1):
        current = np.arange(3) == color
        # 每种颜色按真实拿起、画完、归还推进；跨颜色的顺序完全由调用者决定。
        reward += progress.advance(current, empty, empty)
        reward += progress.advance(current, current, empty)
        reward += progress.advance(empty, current, current)
        assert progress.picked.sum() == progress.returned.sum() == count
        # 同一物理状态重复检查不能重复领取抓取或归还奖励。
        assert progress.advance(empty, current, current) == 0
    assert progress.returned.all() and reward == pytest.approx(0.6)


def test_return_requires_a_completed_active_stroke() -> None:
    # 只移动笔、只显示已画图或碰到非任务颜色，都不能冒充完成整条线。
    # 单色任务保留另外两支物理笔，用于检查active mask。
    task = generate_task(17, 1)
    progress = PaintProgress(task)
    active = np.arange(3) == task.strokes[0].color
    empty = np.zeros(3, dtype=bool)
    # 没有实际抓起时，画面和归位信号不构成有效示范。
    assert progress.advance(empty, active, active) == 0
    assert not progress.returned.any()
    # 非任务笔即使完整走过所有状态也不能贡献奖励。
    assert progress.advance(~active, ~active, ~active) == 0
    # 抓起后没有画完就放回，只有抓取奖励，没有完成奖励。
    assert progress.advance(active, empty, empty) == pytest.approx(0.1)
    assert progress.advance(empty, empty, active) == 0
    assert not progress.returned.any()
