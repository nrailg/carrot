"""Runtime interfaces used by the controller layer."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Protocol

from carrot.distributed.worker import Worker


class Future[T](Protocol):
    def result(self, timeout: float | None = None) -> T: ...

    def done(self) -> bool: ...

    def cancel(self) -> bool: ...


class ActorHandle(Protocol):
    @property
    def rank(self) -> int: ...

    def call(self, method: str, *args: Any, **kwargs: Any) -> Future[Any]: ...


@dataclass(frozen=True)
class WorkerSpec:
    """Serializable recipe for constructing a worker in its target process."""

    cls: type[Worker]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def build(self) -> Worker:
        worker = self.cls(*self.args, **dict(self.kwargs))
        assert isinstance(worker, Worker), f"{self.cls.__name__} must be a Worker"
        return worker


@dataclass(frozen=True)
class Channel[T]:
    """Serializable bounded FIFO used between workers."""

    name: str
    _queue: Any = field(repr=False)

    def put(self, item: T, timeout: float | None = None) -> None:
        self._queue.put(item, block=True, timeout=timeout)

    def get(self, timeout: float | None = None) -> T:
        return self._queue.get(block=True, timeout=timeout)

    def get_batch(
        self,
        max_items: int,
        *,
        min_items: int = 1,
        timeout: float | None = None,
    ) -> list[T]:
        assert 1 <= min_items <= max_items, "expected 1 <= min_items <= max_items"
        deadline = None if timeout is None else time.monotonic() + timeout
        items = []
        while len(items) < min_items:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            items.append(self.get(timeout=remaining))
        while len(items) < max_items:
            try:
                items.append(self._queue.get_nowait())
            except Empty:
                break
        return items

    def empty(self) -> bool:
        return self._queue.empty()
