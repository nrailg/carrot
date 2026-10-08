import json
import logging
import platform
import time
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from PIL import Image

from .actions import ActionSink, LogSink, SO101Sink, action_limits
from .client import PolicyClient
from .config import JOINT_NAMES, DeploymentConfig
from .dataset import load_dataset_source
from .observations import ObservationSource, RobotSource
from .report import write_report


class RunLog:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.events = (directory / "events.jsonl").open("x", buffering=1)

    def write(self, event: str, **values: Any) -> None:
        self.events.write(json.dumps({"event": event, "time": time.time(), **values},
                                     allow_nan=False) + "\n")

    def close(self) -> None:
        self.events.close()


def validate_metadata(metadata: dict, config: DeploymentConfig) -> int:
    assert metadata["embodiment"] == "so101", "server must serve a SO101 policy"
    assert metadata["action_dim"] == 6, "server must return six-dimensional actions"
    horizon = metadata["action_horizon"]
    assert type(horizon) is int and horizon >= config.execute_steps >= 1, "invalid action horizon"
    assert type(metadata["num_steps"]) is int and metadata["num_steps"] >= 1
    return horizon


def validate_actions(response: dict, horizon: int) -> np.ndarray:
    actions = np.asarray(response["actions"])
    assert actions.dtype == np.float32 and actions.shape == (horizon, 6), (
        f"expected float32[{horizon},6], got {actions.dtype}{actions.shape}"
    )
    assert np.isfinite(actions).all(), "non-finite policy actions"
    return actions


def run_loop(config: DeploymentConfig, source: ObservationSource, sink: ActionSink,
             policy: PolicyClient, log: RunLog) -> dict[str, Any]:
    """Consume observation chunks synchronously, stopping at an episode boundary.

    Parameters
    ----------
    config : DeploymentConfig
    source : ObservationSource
    sink : ActionSink
    policy : PolicyClient
    log : RunLog
        Owned by the caller; events are flushed after each request and action.

    Returns
    -------
    dict
        Completion counts and reason. Errors propagate without another action being sent.
    """
    horizon = validate_metadata(policy.metadata, config)
    frame = source.read()
    assert frame is not None, "observation source is empty"
    for key, name in config.image_keys.items():
        Image.fromarray(frame.request[key]).save(log.directory / f"first_{name}.png")
    log.write("metadata", metadata=policy.metadata)

    # 预热只验证首个响应，不把返回动作交给 sink。
    validate_actions(policy.infer(frame.request, timeout=config.warmup_timeout_s), horizon)
    log.write("warmup_complete")

    if config.observation_source == "dataset":
        # 数据集动作若连续下发实机，需先核对录制起点与实机起点。
        multi_step = config.execute_steps > 1 or config.max_chunks != 1
        initial = sink.check_initial(frame.request["observation/state"], multi_step=multi_step)
        log.write("initial_state", recorded=frame.request["observation/state"].tolist(), **initial)
    else:
        # 预热期间机器人可能移动，正式推理前重新采集观测。
        frame = source.read()

    chunks = 0
    steps = 0
    while frame is not None:
        start = time.monotonic()
        log.write("request", chunk=chunks, episode=frame.episode, frame=frame.frame,
                  state=frame.request["observation/state"].tolist(), prompt=frame.request["prompt"])
        response = policy.infer(frame.request)
        actions = validate_actions(response, horizon)
        elapsed = time.monotonic() - start
        assert elapsed <= config.request_timeout_s, "policy response exceeded request deadline"

        # 每块仅消费前 count 行，且不得超过数据集 episode 的有效帧。
        count = min(config.execute_steps, frame.valid_steps)
        assert count > 0, "no valid actions remaining"

        arrays = {"predicted": actions, "state": frame.request["observation/state"],
                  "frame": np.array(frame.frame), "planned_steps": np.array(count)}
        if frame.reference is not None:
            arrays["reference"] = frame.reference
        np.savez_compressed(log.directory / f"chunk_{chunks:06d}.npz", **arrays)
        log.write("prediction", chunk=chunks, infer_ms=elapsed * 1000,
                  consume=count, valid_steps=frame.valid_steps)

        for index in range(count):
            tick = time.monotonic()
            log.write("command", chunk=chunks, index=index, target=actions[index].tolist())
            result = sink.send(actions[index])
            steps += 1
            log.write("action", chunk=chunks, index=index, **result)
            # 超时动作已经发送，先留存动作与反馈，再停止下一次推理。
            if config.wait_for_target:
                assert result["target_reached"], (
                    f"robot target timeout after {result['target_wait_s']:.3f}s; "
                    f"residual: {result['target_error']}"
                )
            delay = 1 / config.fps - (time.monotonic() - tick)
            if delay > 0:
                time.sleep(delay)

        source.advance(count)
        chunks += 1
        if config.max_chunks is not None and chunks >= config.max_chunks:
            return {"reason": "max_chunks", "chunks": chunks, "steps": steps}
        frame = source.read()
    return {"reason": "episode_end", "chunks": chunks, "steps": steps}


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
        cameras=cameras, use_degrees=config.use_degrees,
        max_relative_target=float(config.max_relative_target),
        disable_torque_on_disconnect=False,
    ))
    assert robot.calibration, "existing robot calibration must be nonempty"
    stack.callback(close_robot, robot)
    robot.connect(calibrate=False)
    assert robot.is_calibrated, "device calibration differs from the existing calibration file"
    if config.action_sink == "robot":
        # 在 SDK 默认配置之后覆盖，避免重连将这两轴的 P 重设为 16。
        for joint in ("elbow_flex", "wrist_flex"):
            robot.bus.write("Lock", joint, 0, normalize=False, num_retry=2)
            try:
                robot.bus.write("P_Coefficient", joint, 32, normalize=False, num_retry=2)
                actual = robot.bus.read("P_Coefficient", joint, normalize=False, num_retry=2)
                assert actual == 32, f"{joint} P_Coefficient readback mismatch: {actual}"
            finally:
                robot.bus.write("Lock", joint, 1, normalize=False, num_retry=2)
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
            horizon = validate_metadata(client.metadata, config)

            source = None
            if config.observation_source == "dataset":
                source = load_dataset_source(config, horizon)
            robot = None
            if config.observation_source == "robot" or config.action_sink == "robot":
                robot = connect_robot(config, stack)
            if config.observation_source == "robot":
                source = RobotSource(robot, config, horizon)
            assert source is not None

            if config.action_sink == "log":
                sink = LogSink()
            else:
                lower, upper = action_limits(robot)
                log.write("action_limits", joints=JOINT_NAMES,
                          lower=lower.tolist(), upper=upper.tolist())
                sink = SO101Sink(robot, config, lower, upper)
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
