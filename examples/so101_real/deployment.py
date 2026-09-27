import json
import logging
import platform
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from .actions import LogSink, SO101Sink
from .client import PolicyClient
from .config import DeploymentConfig
from .dataset import load_dataset_source
from .observations import RobotSource
from .report import write_report
from .runtime import RunLog, run_loop, validate_metadata


def close_robot(robot: SO101Follower) -> None:
    try:
        # 退出时保留舵机扭矩；相机仍需在总线关闭失败时逐个断开。
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
    finally:
        for camera in robot.cameras.values():
            if camera.is_connected:
                camera.disconnect()


def connect_robot(config: DeploymentConfig, stack: ExitStack) -> SO101Follower:
    cameras = {}
    if config.observation_source == "robot":
        cameras = {name: OpenCVCameraConfig(**values) for name, values in config.cameras.items()}
        for camera in cameras.values():
            assert camera.color_mode.value == "rgb", "configure RGB camera output"
            assert camera.fps == config.fps, "camera FPS must match deployment"
    robot = SO101Follower(SO101FollowerConfig(
        port=config.robot_port, id=config.robot_id, calibration_dir=Path(config.calibration_dir),
        cameras=cameras, use_degrees=False, max_relative_target=float(config.max_relative_target),
        disable_torque_on_disconnect=False,
    ))
    assert robot.calibration, "existing robot calibration must be nonempty"
    stack.callback(close_robot, robot)
    robot.connect(calibrate=False)
    assert robot.is_calibrated, "device calibration differs from the existing calibration file"
    return robot


def run(config: DeploymentConfig) -> None:
    """Run one deployment session and close network/hardware on every exit.

    Parameters
    ----------
    config : DeploymentConfig
        output_dir must not already exist; hardware is opened only for robot modes.

    Raises
    ------
    AssertionError
        If the configured dataset, policy or robot contract is invalid.
    """
    config.validate()
    directory = Path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=False)

    # 在连接服务或硬件前留存配置与源码指纹，便于核对本次运行来源。
    (directory / "config.json").write_text(json.dumps(asdict(config), indent=2))
    versions = {name: version(name) for name in ("lerobot", "numpy", "openpi-client", "websockets")}
    versions["python"] = platform.python_version()
    versions["client_sources_sha256"] = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    if config.observation_source == "dataset":
        info = Path(config.dataset_root) / "meta" / "info.json"
        versions["dataset_info_sha256"] = sha256(info.read_bytes()).hexdigest()
    if config.observation_source == "robot" or config.action_sink == "robot":
        calibration = Path(config.calibration_dir) / f"{config.robot_id}.json"
        versions["calibration_sha256"] = sha256(calibration.read_bytes()).hexdigest()
    (directory / "versions.json").write_text(json.dumps(versions, indent=2))

    with ExitStack() as stack:
        log = RunLog(directory)
        stack.callback(log.close)
        try:
            client = PolicyClient(config.server_uri, config.connect_timeout_s,
                                  config.request_timeout_s)
            stack.callback(client.close)
            horizon = validate_metadata(client.metadata, config.execute_steps)

            source = None
            if config.observation_source == "dataset":
                source = load_dataset_source(config, horizon)
            robot = None
            if config.observation_source == "robot" or config.action_sink == "robot":
                robot = connect_robot(config, stack)
            if config.observation_source == "robot":
                source = RobotSource(robot, config.prompt, horizon)
            assert source is not None

            sink = LogSink() if config.action_sink == "log" else SO101Sink(
                robot, config.initial_state_tolerance
            )
            summary = run_loop(config, source, sink, client, log)
        except BaseException as error:
            # 异常也写入持久化记录，ExitStack 随后关闭日志、网络和硬件。
            log.write("stopped", reason=type(error).__name__, detail=str(error))
            (directory / "summary.json").write_text(json.dumps({
                "reason": type(error).__name__, "detail": str(error), "completed": False,
            }, indent=2))
            raise
        else:
            log.write("complete", **summary)
            (directory / "summary.json").write_text(json.dumps({
                **summary, "completed": True,
            }, indent=2))

    write_report(directory)
    logging.info("Run saved to %s", directory)
