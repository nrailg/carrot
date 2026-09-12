"""Simulator sidecars that keep robot SDKs out of the core runtime venv."""

from carrot.sim.sidecar import (
    PROTOCOL_VERSION,
    SIDECAR_SERVE_SOURCE,
    SimulatorSidecar,
    SimulatorSupervisor,
)

__all__ = [
    "PROTOCOL_VERSION",
    "SIDECAR_SERVE_SOURCE",
    "SimulatorSidecar",
    "SimulatorSupervisor",
]
