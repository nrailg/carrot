"""Controller-side cluster facade."""

from __future__ import annotations

from typing import Any

from carrot.distributed.group import WorkerGroup
from carrot.distributed.placement import PlacementSpec, RolePlacement
from carrot.distributed.ray_runtime import RayRuntime
from carrot.distributed.runtime import Channel, WorkerSpec


class Cluster:
    """Own the hidden Ray runtime and launched worker groups."""

    def __init__(
        self,
        *,
        address: str | None = None,
        namespace: str = "carrot",
        env_vars: dict[str, str] | None = None,
    ) -> None:
        self._runtime = RayRuntime(
            address=address,
            namespace=namespace,
            env_vars=env_vars,
        )
        self._groups: dict[str, WorkerGroup] = {}
        self._closed = False

    def reserve(self, name: str, spec: PlacementSpec) -> None:
        """Reserve a named Ray placement group that roles may share."""
        assert not self._closed, "cluster is closed"
        self._runtime.reserve(name, spec)

    def launch(
        self,
        name: str,
        worker_cls: type,
        *args: Any,
        placement: RolePlacement | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> WorkerGroup:
        assert not self._closed, "cluster is closed"
        assert name, "worker group name cannot be empty"
        assert name not in self._groups, f"worker group {name!r} already exists"

        if placement is None:
            pool_name = f"__{name}"
            self._runtime.reserve(pool_name, PlacementSpec())
            placement = RolePlacement(pool=pool_name)
        spec = WorkerSpec(worker_cls, args, kwargs)
        workers = self._runtime.launch_group(
            spec,
            placement,
            name,
            env_vars or {},
        )
        group = WorkerGroup(name, workers)
        self._groups[name] = group
        return group

    def channel(self, name: str, maxsize: int = 0) -> Channel[Any]:
        assert not self._closed, "cluster is closed"
        return self._runtime.channel(name, maxsize)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._groups.clear()
        self._runtime.close()

    def __enter__(self) -> Cluster:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
