"""Backend-neutral distributed programming primitives."""

from carrot.distributed.cluster import Cluster
from carrot.distributed.group import (
    GroupResult,
    RankCall,
    WorkerExecutionError,
    WorkerGroup,
    WorkerGroupView,
)
from carrot.distributed.placement import PlacementSpec, RolePlacement
from carrot.distributed.runtime import Channel, WorkerSpec
from carrot.distributed.worker import Worker

__all__ = [
    "Channel",
    "Cluster",
    "GroupResult",
    "PlacementSpec",
    "RankCall",
    "RolePlacement",
    "Worker",
    "WorkerExecutionError",
    "WorkerGroup",
    "WorkerGroupView",
    "WorkerSpec",
]
