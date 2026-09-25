"""Resolve user-configured dataset integrations."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def load_callable(path: str) -> Callable[..., Any]:
    """Import a callable from a ``package.module.symbol`` path.

    Parameters
    ----------
    path : str
        Fully qualified callable path.

    Returns
    -------
    collections.abc.Callable

    Raises
    ------
    AssertionError
        If `path` does not name a module attribute or is not callable.
    """
    module_path, separator, name = path.rpartition(".")
    assert separator and module_path and name, (
        f"callable path must be package.module.symbol, got {path!r}"
    )
    module = importlib.import_module(module_path)
    symbol = vars(module).get(name)
    assert callable(symbol), f"{path!r} must resolve to a callable"
    return symbol
