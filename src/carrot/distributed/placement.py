"""User-facing resource pool and role placement declarations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlacementSpec:
    """Declare the bundles reserved by one Ray placement group.

    Bundle ranks are logical: Carrot probes Ray's allocation and sorts bundles
    by node IP and physical GPU ID before a role selects them.
    """

    num_nodes: int = 1
    bundles_per_node: int = 1
    cpus_per_bundle: float = 1
    gpus_per_bundle: float = 0
    strategy: str = "PACK"

    def __post_init__(self) -> None:
        assert self.num_nodes >= 1, "num_nodes must be positive"
        assert self.bundles_per_node >= 1, "bundles_per_node must be positive"
        assert self.cpus_per_bundle >= 0, "cpus_per_bundle cannot be negative"
        assert self.gpus_per_bundle >= 0, "gpus_per_bundle cannot be negative"
        assert self.strategy in {"PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"}, (
            f"unsupported Ray placement strategy {self.strategy!r}"
        )

    @property
    def num_bundles(self) -> int:
        return self.num_nodes * self.bundles_per_node


@dataclass(frozen=True)
class RolePlacement:
    """Place one role's actors on selected logical bundles."""

    pool: str
    bundle_ranks: tuple[int, ...] | None = None
    cpus_per_actor: float = 1
    gpus_per_actor: float = 0

    def __post_init__(self) -> None:
        assert self.pool, "pool name cannot be empty"
        if self.bundle_ranks is not None:
            assert self.bundle_ranks, "bundle_ranks cannot be empty"
            assert all(rank >= 0 for rank in self.bundle_ranks), "bundle ranks cannot be negative"
        assert self.cpus_per_actor >= 0, "cpus_per_actor cannot be negative"
        assert self.gpus_per_actor >= 0, "gpus_per_actor cannot be negative"
