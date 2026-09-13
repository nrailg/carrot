from __future__ import annotations

import pytest

from carrot.data.loading import load_callable
from carrot.data.lerobot import robotwin_preprocess


def test_load_callable_resolves_default_robotwin_preprocess() -> None:
    assert load_callable("carrot.data.lerobot.robotwin_preprocess") is robotwin_preprocess


def test_load_callable_rejects_non_qualified_path() -> None:
    with pytest.raises(ValueError, match="package.module.symbol"):
        load_callable("robotwin_preprocess")
