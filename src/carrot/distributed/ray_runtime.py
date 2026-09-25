"""Private Ray integration for Carrot's distributed API."""

from __future__ import annotations

import os
import socket
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.queue import Queue
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from carrot.distributed.placement import PlacementSpec, RolePlacement
from carrot.distributed.runtime import ActorHandle, Channel, WorkerSpec

NOSET_VISIBLE_DEVICES_ENV_VARS = (
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
)


@dataclass
class _ReservedPool:
    spec: PlacementSpec
    placement_group: Any
    bundles: list[tuple[int, str, list[int | float]]]
    used_cpus: list[float] = field(default_factory=list)
    used_gpus: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.used_cpus = [0.0] * self.spec.num_bundles
        self.used_gpus = [0.0] * self.spec.num_bundles


def _resolve_visible_device_id(physical_device_id: int | float | str) -> int:
    raw_value = str(physical_device_id).strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return int(float(raw_value))

    ids = [item.strip() for item in visible.split(",") if item.strip()]
    if raw_value in ids:
        return ids.index(raw_value)
    value = int(float(raw_value))
    if str(value) in ids:
        return ids.index(str(value))
    assert 0 <= value < len(ids), (
        f"device {raw_value} is invalid under CUDA_VISIBLE_DEVICES={visible}; "
        f"expected one of {ids} or a local index"
    )
    return value


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


@ray.remote
class _InfoActor:
    def location(self) -> tuple[str, list[int | float]]:
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()


@ray.remote
class _RayActor:
    def __init__(
        self,
        spec: WorkerSpec,
        env_vars: Mapping[str, str],
        master_addr: str | None,
        master_port: int | None,
    ) -> None:
        os.environ.update(env_vars)
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            os.environ["LOCAL_RANK"] = str(_resolve_visible_device_id(gpu_ids[0]))
        self._master_addr = master_addr or ray.util.get_node_ip_address()
        self._master_port = master_port or _free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        self._worker = spec.build()

    def master_addr_and_port(self) -> tuple[str, int]:
        return self._master_addr, self._master_port

    def setup(self) -> None:
        self._worker.setup()

    def call(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # WorkerGroup dispatches methods by name.
        target = getattr(self._worker, method)
        assert callable(target), f"worker attribute {method!r} is not callable"
        return target(*args, **kwargs)

    def teardown(self) -> None:
        self._worker.teardown()


class RayFuture:
    def __init__(self, ref: ray.ObjectRef) -> None:
        self._ref = ref

    def result(self, timeout: float | None = None) -> Any:
        return ray.get(self._ref, timeout=timeout)

    def done(self) -> bool:
        ready, _ = ray.wait([self._ref], timeout=0)
        return bool(ready)

    def cancel(self) -> bool:
        if self.done():
            return False
        ray.cancel(self._ref)
        return True


class RayActorHandle(ActorHandle):
    def __init__(self, rank: int, actor: Any) -> None:
        self._rank = rank
        self._actor = actor

    @property
    def rank(self) -> int:
        return self._rank

    def call(self, method: str, *args: Any, **kwargs: Any) -> RayFuture:
        return RayFuture(self._actor.call.remote(method, args, kwargs))


class RayRuntime:
    """Own Ray resources while keeping Ray types out of the public API."""

    def __init__(
        self,
        *,
        address: str | None = None,
        namespace: str = "carrot",
        env_vars: Mapping[str, str] | None = None,
    ) -> None:
        self._owns_ray = not ray.is_initialized()
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        self._runtime_env_vars = {
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS},
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
            **(env_vars or {}),
        }
        if self._owns_ray:
            ray.init(
                address=address,
                namespace=namespace,
                runtime_env={"env_vars": self._runtime_env_vars},
            )
        self._pools: dict[str, _ReservedPool] = {}
        self._groups: dict[str, list[RayActorHandle]] = {}
        self._channels: dict[str, Channel[Any]] = {}
        self._closed = False

    def reserve(self, name: str, spec: PlacementSpec) -> None:
        assert not self._closed, "runtime is closed"
        assert name not in self._pools, f"placement pool {name!r} already exists"

        nodes = self._selected_nodes(spec.num_nodes)
        bundles = []
        for node in nodes:
            for _ in range(spec.bundles_per_node):
                bundle = {
                    "CPU": spec.cpus_per_bundle,
                    f"node:{node['NodeManagerAddress']}": 0.001,
                }
                if spec.gpus_per_bundle:
                    bundle["GPU"] = spec.gpus_per_bundle
                bundles.append(bundle)

        pg = placement_group(bundles, strategy=spec.strategy)
        ray.get(pg.ready())
        self._pools[name] = _ReservedPool(
            spec=spec,
            placement_group=pg,
            bundles=self._sorted_bundles(pg, spec),
        )

    def launch_group(
        self,
        spec: WorkerSpec,
        placement: RolePlacement,
        group_name: str,
        env_vars: Mapping[str, str],
    ) -> list[RayActorHandle]:
        assert not self._closed, "runtime is closed"
        assert group_name not in self._groups, f"worker group {group_name!r} already exists"
        assert placement.pool in self._pools, f"unknown placement pool {placement.pool!r}"
        pool = self._pools[placement.pool]

        bundle_ranks = placement.bundle_ranks or tuple(range(pool.spec.num_bundles))
        self._validate_role_resources(pool, bundle_ranks, placement)
        selected_bundles = [pool.bundles[rank] for rank in bundle_ranks]
        local_world_sizes = Counter(node_ip for _, node_ip, _ in selected_bundles)
        local_ranks: Counter[str] = Counter()
        node_ranks = {ip: rank for rank, ip in enumerate(sorted(local_world_sizes))}

        handles = []
        master_addr = None
        master_port = None
        try:
            for rank, (bundle_index, node_ip, _) in enumerate(selected_bundles):
                actor_env = dict(self._runtime_env_vars)
                actor_env.update(env_vars)
                actor_env.update(
                    {
                        "GROUP_NAME": group_name,
                        "RANK": str(rank),
                        "WORLD_SIZE": str(len(bundle_ranks)),
                        "LOCAL_RANK": str(local_ranks[node_ip]),
                        "LOCAL_WORLD_SIZE": str(local_world_sizes[node_ip]),
                        "NODE_RANK": str(node_ranks[node_ip]),
                    }
                )
                actor = _RayActor.options(
                    num_cpus=placement.cpus_per_actor,
                    num_gpus=placement.gpus_per_actor,
                    runtime_env={"env_vars": actor_env},
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pool.placement_group,
                        placement_group_bundle_index=bundle_index,
                    ),
                ).remote(spec, actor_env, master_addr, master_port)
                handle = RayActorHandle(rank, actor)
                handles.append(handle)
                local_ranks[node_ip] += 1
                if rank == 0:
                    master_addr, master_port = ray.get(actor.master_addr_and_port.remote())

            ray.get([handle._actor.setup.remote() for handle in handles])
        except BaseException:
            for handle in reversed(handles):
                ray.kill(handle._actor, no_restart=True)
            raise

        for bundle_rank in bundle_ranks:
            pool.used_cpus[bundle_rank] += placement.cpus_per_actor
            pool.used_gpus[bundle_rank] += placement.gpus_per_actor
        self._groups[group_name] = handles
        return handles

    @staticmethod
    def _validate_role_resources(
        pool: _ReservedPool,
        bundle_ranks: tuple[int, ...],
        placement: RolePlacement,
    ) -> None:
        invalid = [rank for rank in bundle_ranks if rank >= pool.spec.num_bundles]
        assert not invalid, (
            f"bundle ranks {invalid} are outside pool {placement.pool!r} "
            f"with {pool.spec.num_bundles} bundles"
        )
        for bundle_rank, actor_count in Counter(bundle_ranks).items():
            requested_cpus = actor_count * placement.cpus_per_actor
            requested_gpus = actor_count * placement.gpus_per_actor
            assert (
                pool.used_cpus[bundle_rank] + requested_cpus <= pool.spec.cpus_per_bundle
            ), f"CPU capacity exceeded on {placement.pool}[{bundle_rank}]"
            assert (
                pool.used_gpus[bundle_rank] + requested_gpus <= pool.spec.gpus_per_bundle
            ), f"GPU capacity exceeded on {placement.pool}[{bundle_rank}]"

    def channel(self, name: str, maxsize: int = 0) -> Channel[Any]:
        assert not self._closed, "runtime is closed"
        assert name not in self._channels, f"channel {name!r} already exists"
        channel = Channel(name=name, _queue=Queue(maxsize=maxsize))
        self._channels[name] = channel
        return channel

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        handles = [handle for group in self._groups.values() for handle in group]
        close_refs = [handle._actor.teardown.remote() for handle in handles]
        if close_refs:
            try:
                ray.get(close_refs)
            except Exception as error:
                errors.append(error)
        for handle in handles:
            ray.kill(handle._actor, no_restart=True)
        self._groups.clear()
        for channel in self._channels.values():
            channel._queue.shutdown(force=True)
        self._channels.clear()
        for pool in self._pools.values():
            remove_placement_group(pool.placement_group)
        self._pools.clear()
        if self._owns_ray:
            ray.shutdown()
        if errors:
            raise RuntimeError(f"{len(errors)} worker(s) failed during shutdown") from errors[0]

    @staticmethod
    def _selected_nodes(num_nodes: int) -> list[dict[str, Any]]:
        nodes = [node for node in ray.nodes() if node["Alive"]]
        nodes.sort(key=lambda node: (node["NodeManagerAddress"], node["NodeID"]))
        assert len(nodes) >= num_nodes, (
            f"requested {num_nodes} Ray nodes, but only {len(nodes)} are alive"
        )
        return nodes[:num_nodes]

    @staticmethod
    def _sorted_bundles(
        pg: Any,
        placement: PlacementSpec,
    ) -> list[tuple[int, str, list[int | float]]]:
        actors = []
        for bundle_index in range(placement.num_bundles):
            actor = _InfoActor.options(
                num_cpus=placement.cpus_per_bundle,
                num_gpus=placement.gpus_per_bundle,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_index,
                ),
            ).remote()
            actors.append(actor)
        locations = ray.get([actor.location.remote() for actor in actors])
        for actor in actors:
            ray.kill(actor)
        return sorted(
            [
                (bundle_index, node_ip, gpu_ids)
                for bundle_index, (node_ip, gpu_ids) in enumerate(locations)
            ],
            key=lambda item: (
                item[1],
                item[2][0] if item[2] else item[0],
            ),
        )
