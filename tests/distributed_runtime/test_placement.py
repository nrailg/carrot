import pytest

from carrot.distributed import PlacementSpec, RolePlacement


def test_placement_spec_computes_world_size() -> None:
    placement = PlacementSpec(num_nodes=2, bundles_per_node=4, gpus_per_bundle=1)

    assert placement.num_bundles == 8


@pytest.mark.parametrize("field", ["num_nodes", "bundles_per_node"])
def test_placement_spec_rejects_non_positive_counts(field: str) -> None:
    # 验证节点数或每节点 bundle 数为零时，用户配置校验触发带消息断言。
    values = {"num_nodes": 1, "bundles_per_node": 1}
    values[field] = 0

    with pytest.raises(AssertionError, match="must be positive"):
        PlacementSpec(**values)


def test_role_placement_accepts_repeated_bundles_for_multiple_actors() -> None:
    placement = RolePlacement(
        pool="sim",
        bundle_ranks=(0, 0, 1, 1),
        cpus_per_actor=0.5,
    )

    assert placement.bundle_ranks == (0, 0, 1, 1)
