"""Minimal proof that Ray supervisors can drive simulators in separate venvs."""

from __future__ import annotations

import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

from carrot.distributed import Cluster, PlacementSpec, RolePlacement
from carrot.sim import PROTOCOL_VERSION, SIDECAR_SERVE_SOURCE, SimulatorSupervisor

_SIDECAR_PROGRAM = SIDECAR_SERVE_SOURCE + f"""
import os
import sys

import carrot_sim_plugin


class PluginEnv:
    def __init__(self) -> None:
        self.backend = carrot_sim_plugin.BACKEND
        if self.backend != os.environ["CARROT_SIM_BACKEND"]:
            raise RuntimeError(f"loaded the wrong simulator plugin: {{self.backend}}")
        self.state = 0

    def health(self) -> dict:
        return {{
            "backend": self.backend,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "plugin_version": carrot_sim_plugin.PLUGIN_VERSION,
            "protocol_version": {PROTOCOL_VERSION},
        }}

    def reset(self, seed: int) -> dict:
        self.state = seed
        return {{"observation": {{"state": self.state}}}}

    def step(self, action: int) -> dict:
        self.state += carrot_sim_plugin.transform_action(action)
        return {{
            "observation": {{"state": self.state}},
            "reward": float(self.state),
            "terminated": False,
            "truncated": False,
        }}

    def close(self) -> None:
        return None


serve(PluginEnv())
"""

_MUJOCO_SIDECAR_PROGRAM = SIDECAR_SERVE_SOURCE + f"""
import sys

import mujoco


class MujocoEnv:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><joint/><geom size='0.1'/></body></worldbody></mujoco>"
        )
        self.data = mujoco.MjData(self.model)

    def health(self) -> dict:
        return {{
            "backend": "mujoco",
            "executable": sys.executable,
            "prefix": sys.prefix,
            "mujoco_version": mujoco.__version__,
            "protocol_version": {PROTOCOL_VERSION},
        }}

    def reset(self, seed: int) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        return {{"time": self.data.time, "nq": self.model.nq}}

    def step(self, action: int) -> dict:
        mujoco.mj_step(self.model, self.data)
        return {{"time": self.data.time, "nq": self.model.nq}}

    def close(self) -> None:
        return None


serve(MujocoEnv())
"""


class SupervisedSimulator(SimulatorSupervisor):
    def supervisor_executable(self) -> str:
        return sys.executable


def _build_plugin_wheel(
    directory: Path,
    *,
    version: str,
    backend: str,
    action_scale: int,
) -> Path:
    """Build a tiny wheel offline so each venv installs a conflicting package version."""
    distribution = "carrot_sim_plugin"
    wheel = directory / f"{distribution}-{version}-py3-none-any.whl"
    package_path = f"{distribution}/__init__.py"
    dist_info = f"{distribution}-{version}.dist-info"
    metadata_path = f"{dist_info}/METADATA"
    wheel_path = f"{dist_info}/WHEEL"
    record_path = f"{dist_info}/RECORD"
    package = (
        f'BACKEND = "{backend}"\n'
        f'PLUGIN_VERSION = "{version}"\n'
        f"ACTION_SCALE = {action_scale}\n\n"
        "def transform_action(action):\n"
        "    return action * ACTION_SCALE\n"
    )
    metadata = f"Metadata-Version: 2.1\nName: carrot-sim-plugin\nVersion: {version}\n"
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        "Generator: carrot-sidecar-test\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    )
    record = "".join(
        f"{path},,\n" for path in (package_path, metadata_path, wheel_path, record_path)
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(package_path, package)
        archive.writestr(metadata_path, metadata)
        archive.writestr(wheel_path, wheel_metadata)
        archive.writestr(record_path, record)
    return wheel


def _create_venv(path: Path, wheel: Path) -> str:
    venv.EnvBuilder(with_pip=True).create(path)
    python = str(path / "bin" / "python")
    subprocess.run(
        [python, "-m", "pip", "install", "--no-index", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    return python


def test_ray_supervises_simulators_in_two_isolated_venvs(tmp_path: Path) -> None:
    libero_venv = tmp_path / "libero-venv"
    maniskill_venv = tmp_path / "maniskill-venv"
    libero_wheel = _build_plugin_wheel(
        tmp_path,
        version="1.0.0",
        backend="libero",
        action_scale=1,
    )
    maniskill_wheel = _build_plugin_wheel(
        tmp_path,
        version="2.0.0",
        backend="maniskill",
        action_scale=2,
    )
    libero_python = _create_venv(libero_venv, libero_wheel)
    maniskill_python = _create_venv(maniskill_venv, maniskill_wheel)
    source_root = Path(__file__).resolve().parents[1] / "src"

    # Ray workers do not inherit pytest's in-process ``pythonpath`` setting.
    # Production images install Carrot; a mounted-source smoke declares it explicitly.
    with Cluster(env_vars={"PYTHONPATH": str(source_root)}) as cluster:
        cluster.reserve("simulators", PlacementSpec(bundles_per_node=2))
        libero = cluster.launch(
            "libero",
            SupervisedSimulator,
            libero_python,
            _SIDECAR_PROGRAM,
            {"CARROT_SIM_BACKEND": "libero"},
            placement=RolePlacement(pool="simulators", bundle_ranks=(0,)),
        )
        maniskill = cluster.launch(
            "maniskill",
            SupervisedSimulator,
            maniskill_python,
            _SIDECAR_PROGRAM,
            {"CARROT_SIM_BACKEND": "maniskill"},
            placement=RolePlacement(pool="simulators", bundle_ranks=(1,)),
        )

        libero_health = libero.call("health").wait()[0]
        maniskill_health = maniskill.call("health").wait()[0]

        assert libero.call("supervisor_executable").wait() == [sys.executable]
        assert maniskill.call("supervisor_executable").wait() == [sys.executable]
        assert libero_health["backend"] == "libero"
        assert maniskill_health["backend"] == "maniskill"
        assert libero_health["plugin_version"] == "1.0.0"
        assert maniskill_health["plugin_version"] == "2.0.0"
        assert libero_health["protocol_version"] == PROTOCOL_VERSION
        assert maniskill_health["protocol_version"] == PROTOCOL_VERSION
        assert Path(libero_health["prefix"]).resolve() == libero_venv.resolve()
        assert Path(maniskill_health["prefix"]).resolve() == maniskill_venv.resolve()
        assert libero_health["prefix"] != maniskill_health["prefix"]

        assert libero.call("reset", 10).wait() == [{"observation": {"state": 10}}]
        assert maniskill.call("reset", 20).wait() == [{"observation": {"state": 20}}]
        assert libero.call("step", 3).wait() == [
            {
                "observation": {"state": 13},
                "reward": 13.0,
                "terminated": False,
                "truncated": False,
            }
        ]
        assert maniskill.call("step", -2).wait() == [
            {
                "observation": {"state": 16},
                "reward": 16.0,
                "terminated": False,
                "truncated": False,
            }
        ]


@pytest.mark.skipif(
    not os.environ.get("CARROT_TEST_MUJOCO_331_PYTHON")
    or not os.environ.get("CARROT_TEST_MUJOCO_312_PYTHON"),
    reason="set paths to prebuilt MuJoCo 3.3.1 and 3.12.0 venv interpreters",
)
def test_ray_supervises_two_real_mujoco_versions() -> None:
    mujoco_331_python = os.environ["CARROT_TEST_MUJOCO_331_PYTHON"]
    mujoco_312_python = os.environ["CARROT_TEST_MUJOCO_312_PYTHON"]
    source_root = Path(__file__).resolve().parents[1] / "src"

    with Cluster(env_vars={"PYTHONPATH": str(source_root)}) as cluster:
        cluster.reserve("mujoco-versions", PlacementSpec(bundles_per_node=2))
        old = cluster.launch(
            "mujoco-331",
            SimulatorSupervisor,
            mujoco_331_python,
            _MUJOCO_SIDECAR_PROGRAM,
            placement=RolePlacement(pool="mujoco-versions", bundle_ranks=(0,)),
        )
        new = cluster.launch(
            "mujoco-312",
            SimulatorSupervisor,
            mujoco_312_python,
            _MUJOCO_SIDECAR_PROGRAM,
            placement=RolePlacement(pool="mujoco-versions", bundle_ranks=(1,)),
        )

        old_health = old.call("health").wait()[0]
        new_health = new.call("health").wait()[0]
        assert old_health["mujoco_version"] == "3.3.1"
        assert new_health["mujoco_version"] == "3.12.0"
        assert old_health["prefix"] != new_health["prefix"]
        assert old.call("reset", 0).wait() == [{"time": 0.0, "nq": 1}]
        assert new.call("reset", 0).wait() == [{"time": 0.0, "nq": 1}]
        assert old.call("step", 0).wait()[0]["time"] > 0
        assert new.call("step", 0).wait()[0]["time"] > 0
