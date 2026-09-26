import json
from pathlib import Path

import numpy as np
import pytest

from examples.so101_real.actions import LogSink, SO101Sink
from examples.so101_real.config import JOINT_NAMES, DeploymentConfig, load_config
from examples.so101_real.observations import DatasetSource, RobotSource
from examples.so101_real.runtime import RunLog, run_loop, validate_metadata


class Dataset:
    def __init__(self, length=7, horizon=4):
        self.length = length
        self.horizon = horizon

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        valid = min(self.horizon, self.length - index)
        return {
            "episode_index": 2, "frame_index": index,
            "observation.state": np.zeros(6, dtype=np.float32),
            "observation.images.top": np.full((3, 12, 16), 123, dtype=np.uint8),
            "observation.images.fpv": np.full((3, 16, 12), 45, dtype=np.uint8),
            "task": "pick orange cube",
            "action": np.full((self.horizon, 6), 99, dtype=np.float32),
            "action_is_pad": np.arange(self.horizon) >= valid,
        }


class Policy:
    metadata = {"embodiment": "so101", "action_dim": 6, "action_horizon": 4, "num_steps": 10}

    def __init__(self):
        self.requests = []

    def infer(self, observation, *, timeout=None):
        self.requests.append(observation)
        return {"actions": np.full((4, 6), 2, dtype=np.float32)}


class Robot:
    def __init__(self):
        self.state = np.zeros(6, dtype=np.float32)
        self.commands = []

    def get_observation(self):
        return {
            **dict(zip(JOINT_NAMES, self.state, strict=True)),
            "top": np.zeros((12, 16, 3), dtype=np.uint8),
            "fpv": np.zeros((16, 12, 3), dtype=np.uint8),
        }

    def send_action(self, action):
        self.commands.append(action)
        target = np.array([action[name] for name in JOINT_NAMES], dtype=np.float32)
        self.state += np.clip(target - self.state, -5, 5)
        return dict(zip(JOINT_NAMES, map(float, self.state), strict=True))


@pytest.fixture
def log(tmp_path, monkeypatch):
    # 只取消节拍等待，保留真实单调时钟用于请求耗时与超时判定。
    monkeypatch.setattr("examples.so101_real.runtime.time.sleep", lambda seconds: None)
    journal = RunLog(tmp_path)
    yield journal
    journal.close()


def test_dataset_mapping_and_episode_boundary(log):
    # 7 帧 episode 按 3 步消费，最后一轮只能消费 1 步；示教动作不能进入请求。
    source = DatasetSource(Dataset(), 2, 0, 4)
    policy = Policy()
    config = DeploymentConfig(execute_steps=3, max_chunks=None)

    # 实际执行调度并读回持久日志，避免只测试函数返回值。
    result = run_loop(config, source, LogSink(), policy, log)
    lines = (log.directory / "events.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    requests = [event for event in events if event["event"] == "request"]

    # 预热不推进帧，末帧不越界，预测动作与值为 99 的示教动作保持隔离。
    assert result == {"reason": "episode_end", "chunks": 3, "steps": 7}
    assert [event["frame"] for event in requests] == [0, 3, 6]
    assert len(policy.requests) == 4
    assert all(set(req) == {"observation/state", "observation/image",
                            "observation/wrist_image", "prompt"} for req in policy.requests)
    assert policy.requests[0]["observation/image"].shape == (12, 16, 3)
    assert np.all(policy.requests[0]["observation/wrist_image"] == 45)
    assert policy.requests[0]["prompt"] == "pick orange cube"
    assert all(event["target"] == [2] * 6 for event in events if event["event"] == "action")
    with np.load(log.directory / "chunk_000002.npz") as chunk:
        assert chunk["reference"].shape == (1, 6)
        assert int(chunk["planned_steps"]) == 1


def test_start_frame_and_prompt_override():
    # 指定起始帧应为 episode 内索引，任务覆盖不改变其余观测。
    source = DatasetSource(Dataset(), 2, 5, 4, "move cube")
    frame = source.read()

    # 末尾两帧可用，示教中的 padding 不作为有效比较样本。
    assert frame.frame == 5 and frame.valid_steps == 2
    assert frame.request["prompt"] == "move cube"
    assert frame.reference.shape == (2, 6)
    with pytest.raises(AssertionError, match="start_frame"):
        DatasetSource(Dataset(), 2, 7, 4)


def test_padding_mismatch_rejected():
    # 错误 padding 会使客户端跨 episode 执行，必须在推理前失败。
    class BadDataset(Dataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["action_is_pad"] = np.ones(4, dtype=bool)
            return sample

    # 对非末尾帧伪造全部 padding，断言不会被当作正常结束吞掉。
    with pytest.raises(AssertionError, match="padding"):
        DatasetSource(BadDataset(), 2, 0, 4).read()


def test_absolute_action_mapping_and_driver_clipping():
    # 绝对目标不加当前状态，日志必须采用驱动限幅后实际下发的值。
    robot = Robot()
    robot.state[:] = 10
    sink = SO101Sink(robot, 10)
    target = np.array([-200, -3, 20, 50, 200, -10], dtype=np.float32)

    # 同时触发合法范围裁剪和驱动的相对目标限幅。
    result = sink.send(target)

    # 固定键序对应五关节加夹爪，实际下发不同于原始预测时必须留证据。
    assert list(robot.commands[0]) == list(JOINT_NAMES)
    assert list(robot.commands[0].values()) == [-100, -3, 20, 50, 100, 0]
    assert result["sent"] == [5, 5, 15, 15, 15, 5]
    assert result["clipped"]
    assert result["present"] == [10] * 6


@pytest.mark.parametrize("execute_steps,max_chunks", [(2, 1), (1, 2), (1, None)])
def test_multi_step_initial_mismatch_stops_before_send(log, execute_steps, max_chunks):
    # 多块单步同样属于多步执行，不能绕过录制状态与真机初始状态的核对。
    robot = Robot()
    robot.state[:] = 30
    config = DeploymentConfig(execute_steps=execute_steps, max_chunks=max_chunks)

    # 预热可以完成，但位置不一致时不得产生任何运动命令。
    with pytest.raises(AssertionError, match="initial state mismatch"):
        run_loop(config, DatasetSource(Dataset(), 2, 0, 4), SO101Sink(robot, 10), Policy(), log)
    assert robot.commands == []


def test_single_action_uses_prediction_not_demonstration(log):
    # 单步动作值取自 policy，预热、示教动作和其余 chunk 都不得下发。
    robot = Robot()
    policy = Policy()
    result = run_loop(DeploymentConfig(), DatasetSource(Dataset(), 2, 0, 4),
                      SO101Sink(robot, 10), policy, log)

    # 只发一条值为 2 的预测；99 是示教动作，不能出现在命令里。
    assert result["steps"] == 1
    assert len(robot.commands) == 1
    assert list(robot.commands[0].values()) == [2] * 6
    assert len(policy.requests) == 2


@pytest.mark.parametrize("failure", [TimeoutError, ConnectionError, KeyboardInterrupt])
def test_failure_never_sends_remaining_actions(log, failure):
    # 第二次正式请求失败时，前一块已完成的动作保留，之后不得继续发送。
    class FailingPolicy(Policy):
        def infer(self, observation, *, timeout=None):
            if len(self.requests) == 2:
                raise failure("injected")
            return super().infer(observation, timeout=timeout)

    robot = Robot()
    config = DeploymentConfig(execute_steps=2, max_chunks=3)

    # 第一块执行两步，然后在下一请求中注入异常并检查命令总数。
    with pytest.raises(failure):
        run_loop(config, DatasetSource(Dataset(), 2, 0, 4),
                 SO101Sink(robot, 10), FailingPolicy(), log)
    assert len(robot.commands) == 2


@pytest.mark.parametrize("actions", [np.zeros((4, 5), dtype=np.float32),
                                    np.full((4, 6), np.nan, dtype=np.float32),
                                    np.zeros((4, 6), dtype=np.float64)])
def test_bad_actions_never_reach_robot(log, actions):
    # shape、数值或 dtype 不符时，连首条动作都不能下发。
    class BadPolicy(Policy):
        def infer(self, observation, *, timeout=None):
            return {"actions": actions}

    robot = Robot()

    # 预热响应也校验完整契约，防止非法响应进入执行循环。
    with pytest.raises(AssertionError):
        run_loop(DeploymentConfig(), DatasetSource(Dataset(), 2, 0, 4),
                 SO101Sink(robot, 10), BadPolicy(), log)
    assert not robot.commands


def test_robot_observations_refresh_after_warmup_and_action(log):
    # 真机源必须在预热后重新观测，并在下一块读到执行后的关节状态。
    robot = Robot()
    policy = Policy()
    config = DeploymentConfig(observation_source="robot", max_chunks=2)

    # 共执行两块，每块一步；真实状态由 fake driver 的实际命令更新。
    run_loop(config, RobotSource(robot, "pick", 4), SO101Sink(robot, 10), policy, log)

    # 第三次请求已包含上一动作的真实反馈，区别于录制观测开环模式。
    assert len(policy.requests) == 3
    np.testing.assert_array_equal(policy.requests[-1]["observation/state"], [2] * 6)


def test_server_metadata_rejects_wrong_embodiment():
    # 六维形状本身不能代表 SO101，握手必须明确 embodiment。
    metadata = {**Policy.metadata, "embodiment": "libero"}

    # 连接错误策略或执行长度超过模型 horizon 时均提前失败。
    with pytest.raises(AssertionError, match="SO101"):
        validate_metadata(metadata, 1)
    with pytest.raises(AssertionError, match="horizon"):
        validate_metadata(Policy.metadata, 5)


def test_default_profile_needs_no_robot_configuration(tmp_path):
    # 默认配置必须是数据集加日志；无需串口、标定或相机，防止意外接入硬件。
    config = load_config(Path("examples/so101_real/deployment.yaml"),
                         {"dataset_root": str(tmp_path)})

    # 真机读写由独立显式选项控制，原始模板保持单块单步。
    assert config.observation_source == "dataset" and config.action_sink == "log"
    assert config.robot_port is None
    assert config.execute_steps == 1 and config.max_chunks == 1
