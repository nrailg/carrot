from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from carrot.data.dataset_spec import SFTDatasetSpec
from examples.so101_real.frame_index import render_frame_index
from carrot.models.pi05.embodiments.so101 import SO101Inputs
from examples.so101_real import runner
from examples.so101_real.actions import LogSink
from examples.so101_real.config import JOINT_NAMES, DeploymentConfig
from examples.so101_real.observations import DatasetSource, RobotSource
from recipes.pi05_sft_so101_arm_dance_frame_index import augmentation
from recipes.pi05_sft_so101_state_jitter.augmentation import StateJitterDataset


class Episode:
    def __init__(self, frames: int = 7, horizon: int = 10) -> None:
        self.frames = frames
        self.horizon = horizon
        self.state = np.arange(6, dtype=np.float32)
        self.actions = np.full((horizon, 6), 42, dtype=np.float32)

    def __len__(self) -> int:
        return self.frames

    def __getitem__(self, index: int) -> dict:
        return {
            "episode_index": 2, "frame_index": index,
            "observation.state": self.state, "action": self.actions,
            "action_is_pad": np.arange(self.horizon) >= min(self.horizon, self.frames - index),
            "task": "Dance with the arm",
            "observation.images.fpv": np.full((12, 16, 3), 17, dtype=np.uint8),
        }


class TrainingSamples(Episode):
    def __getitem__(self, index: int) -> dict:
        return {"observation/state": self.state, "actions": self.actions,
                "prompt": "Dance with the arm", "action_is_pad": np.zeros(10, dtype=bool),
                "observation/wrist_image": np.full((12, 16, 3), 17, dtype=np.uint8)}


class StateOnlyRobot:
    def __init__(self) -> None:
        self.reads = 0

    def get_observation(self) -> dict:
        self.reads += 1
        return dict(zip(JOINT_NAMES, range(6), strict=True))


class Policy:
    metadata = {"embodiment": "so101", "action_dim": 6, "action_horizon": 10, "num_steps": 10}

    def __init__(self) -> None:
        self.requests = []

    def infer(self, observation: dict, *, timeout: float | None = None) -> dict:
        self.requests.append(observation)
        return {"actions": np.zeros((10, 6), dtype=np.float32)}


def test_renderer_contract() -> None:
    # 固定七段编码必须跨调用一致，帧号变化可辨认，边界仍保持RGB像素契约。
    images = [render_frame_index(index) for index in (0, 1, 143, 9999)]

    # 比较精确像素而非容差，防止字体或随机状态影响合成图片。
    assert all(image.shape == (224, 224, 3) and image.dtype == np.uint8 for image in images)
    np.testing.assert_array_equal(images[0], render_frame_index(np.int64(0)))
    assert all(not np.array_equal(a, b) for i, a in enumerate(images) for b in images[i + 1:])
    assert set(np.unique(images[0])) == {0, 255}


@pytest.mark.parametrize("index", [True, np.bool_(False), -1, 10000, 1.5, "1", None])
def test_renderer_rejects_illegal_index(index) -> None:
    # 布尔、非整数与越界值不能被隐式转成可训练帧号。
    with pytest.raises(AssertionError):
        render_frame_index(index)


def test_training_inference_pixels_and_unchanged_targets() -> None:
    # 合成图片只替换腕图；状态增强仍由原wrapper负责，action/夹爪/原样本不得改变。
    raw = TrainingSamples()
    state_before, action_before = raw.state.copy(), raw.actions.copy()
    jittered = StateJitterDataset(raw, 3.0, [[-2, 4]] * 5)
    training = augmentation.FrameIndexDataset(jittered, 7)[3]
    config = DeploymentConfig(base_camera=None, episode=2, frame_index_image_frames=7)
    inference = DatasetSource(Episode(), config, 10).read()
    source = DatasetSource(Episode(), replace(config, start_frame=3), 10)

    # 同一索引调用共享renderer，图像逐元素一致；SO101变换只允许腕部mask为真。
    np.testing.assert_array_equal(training["observation/wrist_image"],
                                  source.read().request["observation/wrist_image"])
    transformed = SO101Inputs()(training)
    assert transformed["image_mask"] == {
        "base_0_rgb": False, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False,
    }
    assert inference.frame == 0

    # 增强仅修改五个角度且遵守共享限位，原始数组和目标引用保留。
    np.testing.assert_array_equal(raw.state, state_before)
    np.testing.assert_array_equal(raw.actions, action_before)
    assert training["actions"] is raw.actions
    assert training["observation/state"][5] == raw.state[5]
    assert np.all(training["observation/state"][:5] >= -2)
    assert np.all(training["observation/state"][:5] <= 4)
    assert abs(training["observation/state"][0] - raw.state[0]) <= 3
    assert np.all(raw[3]["observation/wrist_image"] == 17)


def test_factory_single_episode_and_stats(monkeypatch) -> None:
    # factory必须验证单episode元数据，保留原统计量，不能将跨episode全局索引当进度。
    metadata = Mock(total_episodes=1, total_frames=7)
    spec = SFTDatasetSpec(TrainingSamples(), None, {"state": "unchanged"},
                          {"action": "unchanged"}, ("observation.images.fpv",))
    builder = Mock(return_value=spec)
    monkeypatch.setattr(augmentation, "LeRobotDatasetMetadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(augmentation, "build_jitter_dataset", builder)
    kwargs = dict(repo_id="local/test", root="unused", revision=None,
                  base_image_key=None, wrist_image_key="observation.images.fpv")

    # 只替换dataset对象；复用原factory而非复制训练/归一化逻辑。
    result = augmentation.build_dataset(**kwargs)
    builder.assert_called_once_with(**kwargs)
    assert result.state_stats is spec.state_stats and result.action_stats is spec.action_stats
    assert result.dataset.source is spec.dataset

    # 多episode与长度不符必须失败，不允许静默回退为全局索引。
    metadata.total_episodes = 2
    with pytest.raises(AssertionError, match="exactly one episode"):
        augmentation.build_dataset(**kwargs)
    metadata.total_episodes, metadata.total_frames = 1, 8
    with pytest.raises(AssertionError, match="length"):
        augmentation.build_dataset(**kwargs)


@pytest.mark.parametrize("source_type", ["dataset", "robot"])
@pytest.mark.parametrize("steps", [1, 5])
def test_loop_advances_only_consumed_actions(tmp_path, monkeypatch, source_type, steps) -> None:
    # h10/K1与K5都按已采用动作推进；重复read、预热和等待读状态不能推进。
    config = DeploymentConfig(observation_source=source_type, base_camera=None, episode=2,
                              frame_index_image_frames=7, prompt="Dance with the arm",
                              execute_steps=steps, max_chunks=None)
    robot = StateOnlyRobot()
    source = (DatasetSource(Episode(), config, 10) if source_type == "dataset"
              else RobotSource(robot, config, 10))
    policy = Policy()
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    # 先重复读取模拟预热/等待，不给advance任何机会。
    assert source.read().frame == source.read().frame == 0
    robot.get_observation()
    assert source.read().frame == 0
    log = runner.RunLog(tmp_path)
    try:
        result = runner.run_loop(config, source, LogSink(), policy, log)
    finally:
        log.close()

    # 尾块裁到剩余帧；到N返回None，不能循环或固定最后一张图。
    expected = list(range(0, 7, steps))
    assert result == {"reason": "episode_end", "chunks": len(expected), "steps": 7}
    assert len(policy.requests) == len(expected) + 1
    for request, index in zip(policy.requests, [0, *expected], strict=True):
        np.testing.assert_array_equal(request["observation/wrist_image"], render_frame_index(index))
        assert set(request) == {"observation/state", "observation/wrist_image", "prompt"}
    assert source.frame == 7 and source.read() is None
    with np.load(tmp_path / f"chunk_{len(expected) - 1:06d}.npz") as chunk:
        assert int(chunk["planned_steps"]) == 7 - expected[-1]


def test_start_frame_and_normal_mode() -> None:
    # 合成真机模式尊重显式起点并终止；普通模式仍读取真实腕图且无有限帧限制。
    config = DeploymentConfig(base_camera=None, frame_index_image_frames=7, start_frame=5)
    source = RobotSource(StateOnlyRobot(), config, 10)
    frame = source.read()

    # 末尾仅允许两步，越界advance不能产生最后帧无限重复。
    assert frame.frame == 5 and frame.valid_steps == 2
    with pytest.raises(AssertionError, match="remaining"):
        source.advance(3)
    source.advance(2)
    assert source.read() is None

    # 默认模式保持数据集像素映射，未启用合成时不改变图像与起始行为。
    normal = DatasetSource(Episode(), DeploymentConfig(base_camera=None, episode=2), 10).read()
    assert normal.request["observation/wrist_image"].shape == (12, 16, 3)
    assert np.all(normal.request["observation/wrist_image"] == 17)
    assert RobotSource(StateOnlyRobot(), DeploymentConfig(start_frame=5), 10).frame == 0
    with pytest.raises(AssertionError, match="selected episode length"):
        DatasetSource(Episode(), replace(config, frame_index_image_frames=8), 10)


@pytest.mark.parametrize("changes", [{"frame_index_image_frames": True},
                                     {"frame_index_image_frames": 10001},
                                     {"base_camera": "top"}, {"cameras": {"fpv": {}}},
                                     {"start_frame": 7}])
def test_invalid_synthetic_config(tmp_path, changes) -> None:
    # 配置冲突必须早于任何数据/网络/硬件访问报错。
    config = DeploymentConfig(base_camera=None, dataset_root=str(tmp_path),
                              dataset_repo="local/test", frame_index_image_frames=7)

    # 拒绝非法总帧数、真实相机混用与越界起点。
    with pytest.raises(AssertionError):
        replace(config, **changes).validate()


def test_synthetic_robot_never_constructs_camera(tmp_path, monkeypatch) -> None:
    # 真机合成模式可读实测state，但连接工厂不得构造或打开相机。
    (tmp_path / "arm.json").write_text("{}")
    config = DeploymentConfig(observation_source="robot", base_camera=None,
                              frame_index_image_frames=7, prompt="Dance with the arm",
                              robot_port="fake", robot_id="arm", calibration_dir=str(tmp_path))
    config.validate()
    robot = Mock()
    robot.cameras = {}
    robot.bus.is_connected = True
    built = []
    monkeypatch.setattr(runner, "SO101Follower", lambda cfg: built.append(cfg) or robot)
    camera = Mock(side_effect=AssertionError("camera constructed"))
    monkeypatch.setattr(runner, "OpenCVCameraConfig", camera)

    # 只执行带fake机器人构造器的连接路径，不涉及串口、模型或服务。
    with ExitStack() as stack:
        assert runner.connect_robot(config, stack) is robot

    # SDK收到空camera配置且无相机对象构造，现有清理仍生效。
    assert built[0].cameras == {}
    camera.assert_not_called()
    robot.connect.assert_called_once_with(calibrate=False)
    robot.bus.disconnect.assert_called_once_with(disable_torque=False)
