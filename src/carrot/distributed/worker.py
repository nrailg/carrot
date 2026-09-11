"""Base class for stateful distributed workers."""

from __future__ import annotations

import os


class Worker:
    """A stateful unit of computation hosted by a runtime backend."""

    @property
    def group_name(self) -> str:
        return os.environ["GROUP_NAME"]

    @property
    def rank(self) -> int:
        return int(os.environ["RANK"])

    @property
    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    @property
    def local_rank(self) -> int:
        return int(os.environ["LOCAL_RANK"])

    @property
    def local_world_size(self) -> int:
        return int(os.environ["LOCAL_WORLD_SIZE"])

    @property
    def node_rank(self) -> int:
        return int(os.environ["NODE_RANK"])

    @property
    def master_addr(self) -> str:
        return os.environ["MASTER_ADDR"]

    @property
    def master_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

    def setup(self) -> None:
        """Initialize process-local resources after construction."""

    def teardown(self) -> None:
        """Release process-local resources before worker termination."""
