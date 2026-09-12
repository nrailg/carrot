"""Collective operations over homogeneous stateful workers."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from carrot.distributed.runtime import ActorHandle, Future

T = TypeVar("T")


class WorkerExecutionError(RuntimeError):
    """Adds worker identity to a remote execution failure."""


@dataclass(frozen=True)
class RankCall:
    """Arguments for one rank in an explicit scatter call."""

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None


class GroupResult(Generic[T]):
    """A non-blocking collection of results ordered by global rank."""

    def __init__(
        self,
        method: str,
        calls: Sequence[tuple[int, Future[T]]],
    ) -> None:
        self._method = method
        self._calls = tuple(calls)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(rank for rank, _ in self._calls)

    def done(self) -> bool:
        return all(future.done() for _, future in self._calls)

    def cancel(self) -> bool:
        cancelled = [future.cancel() for _, future in self._calls]
        return all(cancelled)

    def wait(self, timeout: float | None = None) -> list[T]:
        deadline = None if timeout is None else time.monotonic() + timeout
        results: list[T] = []
        for rank, future in self._calls:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                results.append(future.result(timeout=remaining))
            except Exception as error:
                raise WorkerExecutionError(
                    f"worker rank {rank} failed while executing {self._method!r}"
                ) from error
        return results


class WorkerGroupView:
    """Immutable rank selection for one or more collective calls."""

    def __init__(self, group: WorkerGroup, ranks: tuple[int, ...]) -> None:
        self._group = group
        self._ranks = ranks

    def call(self, method: str, *args: Any, **kwargs: Any) -> GroupResult[Any]:
        return self._group.call(method, *args, ranks=self._ranks, **kwargs)

    def map(self, method: str, calls: Sequence[RankCall]) -> GroupResult[Any]:
        return self._group.map(method, calls, ranks=self._ranks)


class WorkerGroup:
    """A runtime-neutral group with broadcast, rank selection, and scatter."""

    def __init__(
        self,
        name: str,
        workers: Iterable[ActorHandle],
    ) -> None:
        self.name = name
        self._workers = tuple(sorted(workers, key=lambda worker: worker.rank))
        expected = tuple(range(len(self._workers)))
        actual = tuple(worker.rank for worker in self._workers)
        if actual != expected:
            raise ValueError(f"worker ranks must be contiguous; expected {expected}, got {actual}")

    @property
    def world_size(self) -> int:
        return len(self._workers)

    def select(self, *ranks: int) -> WorkerGroupView:
        return WorkerGroupView(self, self._normalize_ranks(ranks))

    def call(
        self,
        method: str,
        *args: Any,
        ranks: Iterable[int] | None = None,
        **kwargs: Any,
    ) -> GroupResult[Any]:
        selected = self._normalize_ranks(ranks)
        return GroupResult(
            method,
            [(rank, self._workers[rank].call(method, *args, **kwargs)) for rank in selected],
        )

    def map(
        self,
        method: str,
        calls: Sequence[RankCall],
        *,
        ranks: Iterable[int] | None = None,
    ) -> GroupResult[Any]:
        selected = self._normalize_ranks(ranks)
        if len(calls) != len(selected):
            raise ValueError(f"map received {len(calls)} calls for {len(selected)} selected ranks")
        pending = []
        for rank, call in zip(selected, calls, strict=True):
            pending.append(
                (
                    rank,
                    self._workers[rank].call(
                        method,
                        *call.args,
                        **(call.kwargs or {}),
                    ),
                )
            )
        return GroupResult(method, pending)

    def __iter__(self) -> Iterator[ActorHandle]:
        return iter(self._workers)

    def _normalize_ranks(self, ranks: Iterable[int] | None) -> tuple[int, ...]:
        selected = tuple(range(self.world_size)) if ranks is None else tuple(ranks)
        if len(set(selected)) != len(selected):
            raise ValueError("worker ranks must be unique")
        invalid = [rank for rank in selected if rank < 0 or rank >= self.world_size]
        if invalid:
            raise ValueError(f"invalid worker ranks: {invalid}")
        return selected
