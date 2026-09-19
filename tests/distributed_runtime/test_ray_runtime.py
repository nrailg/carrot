import pytest

from carrot.distributed import (
    Cluster,
    PlacementSpec,
    RankCall,
    RolePlacement,
    Worker,
    WorkerExecutionError,
)


class CounterWorker(Worker):
    def __init__(self, initial: int = 0) -> None:
        super().__init__()
        self.value = initial

    def add(self, amount: int) -> int:
        self.value += amount
        return self.value

    def identity(self) -> tuple[int, int, int, str]:
        return (
            self.rank,
            self.local_rank,
            self.world_size,
            self.group_name,
        )

    def publish(self, channel, value: str) -> int:
        channel.put((self.rank, value))
        return self.rank

    def environment(self, name: str) -> str | None:
        import os

        return os.environ.get(name)

    def fail(self) -> None:
        raise ValueError("worker failure")


@pytest.fixture(scope="module")
def cluster():
    with Cluster() as running_cluster:
        running_cluster.reserve("cpu", PlacementSpec(bundles_per_node=2))
        yield running_cluster


def test_worker_group_is_stateful_and_supports_dispatch(cluster) -> None:
    group = cluster.launch(
        "counter",
        CounterWorker,
        10,
        placement=RolePlacement(pool="cpu", cpus_per_actor=0.1),
        env_vars={"RANK": "99", "NCCL_DEBUG": "INFO"},
    )

    assert group.call("identity").wait() == [
        (0, 0, 2, "counter"),
        (1, 1, 2, "counter"),
    ]
    assert group.call("add", 1).wait() == [11, 11]
    assert group.map("add", [RankCall((2,)), RankCall((3,))]).wait() == [13, 14]
    assert group.select(1).call("add", 5).wait() == [19]
    assert group.call("environment", "NCCL_DEBUG").wait() == ["INFO", "INFO"]


def test_group_result_preserves_rank_and_remote_cause(cluster) -> None:
    group = cluster.launch(
        "failing-counter",
        CounterWorker,
        placement=RolePlacement(pool="cpu", bundle_ranks=(0,), cpus_per_actor=0.1),
    )

    with pytest.raises(WorkerExecutionError, match="worker rank 0") as error:
        group.call("fail").wait()

    assert isinstance(error.value.__cause__, ValueError)


def test_runtime_does_not_rewrite_visible_devices(cluster) -> None:
    group = cluster.launch(
        "visible-env",
        CounterWorker,
        placement=RolePlacement(pool="cpu", cpus_per_actor=0.1),
        env_vars={"CUDA_VISIBLE_DEVICES": "7,9"},
    )

    assert group.call("environment", "CUDA_VISIBLE_DEVICES").wait() == ["7,9", "7,9"]


def test_channel_can_flow_through_workers(cluster) -> None:
    channel = cluster.channel("rollouts", maxsize=2)
    group = cluster.launch(
        "producer",
        CounterWorker,
        placement=RolePlacement(pool="cpu", cpus_per_actor=0.1),
    )

    assert group.call("publish", channel, "trajectory").wait() == [0, 1]
    assert sorted(channel.get_batch(2, min_items=2)) == [
        (0, "trajectory"),
        (1, "trajectory"),
    ]


def test_roles_can_share_or_use_disjoint_bundles(cluster) -> None:
    train = cluster.launch(
        "train",
        CounterWorker,
        placement=RolePlacement(
            pool="cpu",
            bundle_ranks=(0,),
            cpus_per_actor=0.3,
        ),
    )
    generate = cluster.launch(
        "generate",
        CounterWorker,
        placement=RolePlacement(
            pool="cpu",
            bundle_ranks=(0,),
            cpus_per_actor=0.3,
        ),
    )
    with pytest.raises(ValueError, match="CPU capacity exceeded"):
        cluster.launch(
            "overcommitted",
            CounterWorker,
            placement=RolePlacement(
                pool="cpu",
                bundle_ranks=(0,),
                cpus_per_actor=0.1,
            ),
        )
    sim = cluster.launch(
        "sim",
        CounterWorker,
        placement=RolePlacement(
            pool="cpu",
            bundle_ranks=(1,),
            cpus_per_actor=0.5,
        ),
    )

    assert train.call("identity").wait() == [(0, 0, 1, "train")]
    assert generate.call("identity").wait() == [(0, 0, 1, "generate")]
    assert sim.call("identity").wait() == [(0, 0, 1, "sim")]
