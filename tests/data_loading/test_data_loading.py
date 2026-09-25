from __future__ import annotations

import pytest

from carrot.data.loading import load_callable
from carrot.data.robotwin import build_dataset, robotwin_preprocess


def test_load_callable_resolves_default_robotwin_preprocess() -> None:
    # 默认路径须指向 RoboTwin 转换函数，避免训练配置加载失败。
    assert load_callable("carrot.data.robotwin.robotwin_preprocess") is robotwin_preprocess


def test_load_callable_resolves_default_robotwin_dataset() -> None:
    # 数据集工厂迁入 RoboTwin 模块后，配置路径仍须能解析。
    assert load_callable("carrot.data.robotwin.build_dataset") is build_dataset


def test_load_callable_rejects_non_qualified_path() -> None:
    with pytest.raises(ValueError, match="package.module.symbol"):
        load_callable("robotwin_preprocess")
