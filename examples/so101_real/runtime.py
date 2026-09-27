import json
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .actions import ActionSink
from .config import DeploymentConfig
from .observations import ObservationSource


class RemotePolicy(Protocol):
    metadata: dict[str, Any]

    def infer(self, observation: dict[str, Any], *, timeout: float | None = None) -> dict: ...


class RunLog:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.events = (directory / "events.jsonl").open("x", buffering=1)

    def write(self, event: str, **values: Any) -> None:
        self.events.write(json.dumps({"event": event, "time": time.time(), **values},
                                     allow_nan=False) + "\n")

    def close(self) -> None:
        self.events.close()


def validate_metadata(metadata: dict, execute_steps: int) -> int:
    assert metadata["embodiment"] == "so101", "server must serve a SO101 policy"
    assert metadata["action_dim"] == 6, "server must return six-dimensional actions"
    horizon = metadata["action_horizon"]
    assert type(horizon) is int and horizon >= execute_steps >= 1, "invalid action horizon"
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
             policy: RemotePolicy, log: RunLog) -> dict[str, Any]:
    """Consume observation chunks synchronously, stopping at an episode boundary.

    Parameters
    ----------
    config : DeploymentConfig
    source : ObservationSource
    sink : ActionSink
    policy : RemotePolicy
    log : RunLog
        Owned by the caller; events are flushed after each request and action.

    Returns
    -------
    dict
        Completion counts and reason. Errors propagate without another action being sent.
    """
    horizon = validate_metadata(policy.metadata, config.execute_steps)
    frame = source.read()
    assert frame is not None, "observation source is empty"
    for name, key in (("top", "observation/image"), ("fpv", "observation/wrist_image")):
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
            delay = 1 / config.fps - (time.monotonic() - tick)
            if delay > 0:
                time.sleep(delay)

        source.advance(count)
        chunks += 1
        if config.max_chunks is not None and chunks >= config.max_chunks:
            return {"reason": "max_chunks", "chunks": chunks, "steps": steps}
        frame = source.read()
    return {"reason": "episode_end", "chunks": chunks, "steps": steps}
