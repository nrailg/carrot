from contextlib import ExitStack
from dataclasses import replace
from typing import override
from unittest.mock import Mock

import numpy as np
import pytest

from carrot.data.dataset_spec import SFTDatasetSpec
from carrot.models.pi05 import transforms
from carrot.models.pi05.embodiments.so101 import SO101Inputs
from examples.so101_real import runner
from examples.so101_real.actions import LogSink
from examples.so101_real.config import JOINT_NAMES, DeploymentConfig
from examples.so101_real.frame_prompt import format_frame_prompt
from examples.so101_real.observations import DatasetSource, RobotSource
from recipes.pi05_sft_so101_knock_down_the_cylinder_frame_prompt import augmentation, serve
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
            "task": "Knock down the cylinder",
            "observation.images.fpv": np.full((12, 16, 3), 17, dtype=np.uint8),
        }


class TrainingSamples(Episode):
    @override
    def __getitem__(self, index: int) -> dict:
        return {"observation/state": self.state, "actions": self.actions,
                "prompt": "Knock down the cylinder", "action_is_pad": np.zeros(10, dtype=bool),
                "observation/wrist_image": np.full((12, 16, 3), 17, dtype=np.uint8)}


class NoCameraEpisode(Episode):
    @override
    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        del sample["observation.images.fpv"]
        return sample


class StateOnlyRobot:
    def __init__(self) -> None:
        self.reads = 0

    def get_observation(self) -> dict:
        self.reads += 1
        return dict(zip(JOINT_NAMES, range(6), strict=True))


class Policy:
    metadata = {"embodiment": "so101", "action_dim": 6, "action_horizon": 10, "num_steps": 10,
                "frame_index_prompt": True, "no_vision": True}

    def __init__(self) -> None:
        self.requests = []

    def infer(self, observation: dict, *, timeout: float | None = None) -> dict:
        self.requests.append(observation)
        return {"actions": np.zeros((10, 6), dtype=np.float32)}


def test_formatter_contract() -> None:
    # 帧条件只能依赖原始任务与整数索引，尾空白去除规则必须在训练推理间完全相同。
    base = "Knock down the cylinder  "
    values = [format_frame_prompt(base, index) for index in (0, 1, 263, 9999, 10000)]

    # 四位是最小宽度，10000不能截断或因旧图像上限被拒绝；原task重复格式化必须稳定。
    assert values == ["Knock down the cylinder Frame: 0000.",
                      "Knock down the cylinder Frame: 0001.",
                      "Knock down the cylinder Frame: 0263.",
                      "Knock down the cylinder Frame: 9999.",
                      "Knock down the cylinder Frame: 10000."]
    assert values[0] == format_frame_prompt(base, np.int64(0))
    assert base == "Knock down the cylinder  "


@pytest.mark.parametrize("index", [True, np.bool_(False), -1, 1.5, "1", None])
def test_formatter_rejects_illegal_index(index) -> None:
    # 布尔、非整数与负数不能被隐式转成可训练帧号。
    with pytest.raises(AssertionError):
        format_frame_prompt("Knock down the cylinder", index)


@pytest.mark.parametrize("prompt", ["", "  ", None, 1])
def test_formatter_rejects_missing_task(prompt) -> None:
    # 缺失或空任务文本不能被帧号掩盖。
    with pytest.raises(AssertionError, match="task text"):
        format_frame_prompt(prompt, 0)


def test_training_inference_prompt_and_unchanged_targets() -> None:
    # 包装只修改prompt；增强仍由原wrapper负责，原始task/action/夹爪/图像不得改变。
    raw = TrainingSamples()
    state_before, action_before = raw.state.copy(), raw.actions.copy()
    jittered = StateJitterDataset(raw, 3.0, [[-2, 4]] * 5)
    wrapped = augmentation.FramePromptDataset(jittered, 7)
    training = wrapped[3]
    config = DeploymentConfig(base_camera=None, episode=2, frame_index_prompt_frames=7)
    source = DatasetSource(Episode(), replace(config, start_frame=3), 10)

    # 同一索引使用同一formatter；重复read与重新取样不能在已拼接文本上累计suffix。
    assert training["prompt"] == source.read().request["prompt"]
    assert wrapped[3]["prompt"] == source.read().request["prompt"]
    assert training["prompt"] == "Knock down the cylinder Frame: 0003."
    assert raw[3]["prompt"] == "Knock down the cylinder"

    # 增强仅修改五个角度且遵守共享限位，原始数组和图像/目标契约保留。
    np.testing.assert_array_equal(raw.state, state_before)
    np.testing.assert_array_equal(raw.actions, action_before)
    assert training["actions"] is raw.actions
    assert training["observation/state"][5] == raw.state[5]
    assert np.all(training["observation/state"][:5] >= -2)
    assert np.all(training["observation/state"][:5] <= 4)
    assert abs(training["observation/state"][0] - raw.state[0]) <= 3
    assert np.all(training["observation/wrist_image"] == 17)


def test_no_vision_service_uses_shared_drop_vision() -> None:
    # fake policy仅提供输入变换，验证服务与训练共用DropVision，绝不加载或运行模型。
    policy = Mock()
    policy._transform_spec = transforms.Pi05TransformSpec(
        inputs=(SO101Inputs(),), outputs=(), action_dim=6,
    )
    sample = {"observation/state": np.zeros(6, dtype=np.float32),
              "prompt": "Knock down the cylinder Frame: 0003."}

    # 补图仅位于服务适配层，输入原dict与客户端无相机契约不被污染。
    adapted = serve.ZeroWristInput()(sample)
    assert set(sample) == {"observation/state", "prompt"}
    assert adapted["observation/wrist_image"].shape == (224, 224, 3)
    assert adapted["observation/wrist_image"].dtype == np.uint8
    assert not adapted["observation/wrist_image"].any()

    # 使用服务实际完整input_transform验证补图先于SO101Inputs，末端DropVision仍关闭全部mask。
    serve.enable_no_vision(policy)
    transformed = policy._input_transform(sample)

    # 服务接受无图请求，最终所有mask和像素清零，prompt条件保持完整。
    assert transformed["image_mask"] == {
        "base_0_rgb": False, "left_wrist_0_rgb": False, "right_wrist_0_rgb": False,
    }
    assert all(np.all(image == 0) for image in transformed["image"].values())
    assert transformed["prompt"] == sample["prompt"]


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
    config = DeploymentConfig(observation_source=source_type, base_camera=None, wrist_camera=None,
                              episode=2,
                              frame_index_prompt_frames=7, prompt="Knock down the cylinder",
                              execute_steps=steps, max_chunks=None)
    robot = StateOnlyRobot()
    source = (DatasetSource(NoCameraEpisode(), config, 10) if source_type == "dataset"
              else RobotSource(robot, config, 10))
    policy = Policy()
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    # 数据集与机器人替身均无相机字段，重复读取模拟预热/等待不能推进。
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
        assert request["prompt"] == format_frame_prompt("Knock down the cylinder", index)
        assert set(request) == {"observation/state", "prompt"}
    assert source.frame == 7 and source.read() is None
    assert not list(tmp_path.glob("first_*.png"))
    with np.load(tmp_path / f"chunk_{len(expected) - 1:06d}.npz") as chunk:
        assert int(chunk["planned_steps"]) == 7 - expected[-1]


def test_start_frame_and_normal_mode() -> None:
    # 帧prompt真机模式尊重显式起点并终止；普通模式仍读取真实腕图且无有限帧限制。
    config = DeploymentConfig(base_camera=None, wrist_camera=None,
                              frame_index_prompt_frames=7, start_frame=5,
                              prompt="Knock down the cylinder")
    source = RobotSource(StateOnlyRobot(), config, 10)
    frame = source.read()

    # 末尾仅允许两步，越界advance不能产生最后帧无限重复。
    assert frame.frame == 5 and frame.valid_steps == 2
    with pytest.raises(AssertionError, match="remaining"):
        source.advance(3)
    source.advance(2)
    assert source.read() is None

    # 默认模式保持数据集像素映射，未启用帧prompt时不改变图像与起始行为。
    normal = DatasetSource(Episode(), DeploymentConfig(base_camera=None, episode=2), 10).read()
    assert normal.request["observation/wrist_image"].shape == (12, 16, 3)
    assert np.all(normal.request["observation/wrist_image"] == 17)
    assert normal.request["prompt"] == "Knock down the cylinder"
    assert RobotSource(StateOnlyRobot(), DeploymentConfig(start_frame=5), 10).frame == 0
    with pytest.raises(AssertionError, match="selected episode length"):
        DatasetSource(Episode(), replace(config, frame_index_prompt_frames=8), 10)


@pytest.mark.parametrize("changes", [{"frame_index_prompt_frames": True},
                                     {"frame_index_prompt_frames": 0},
                                     {"frame_index_prompt_frames": -1},
                                     {"base_camera": ""}, {"wrist_camera": ""},
                                     {"base_camera": "fpv"}, {"start_frame": 7}])
def test_invalid_frame_prompt_config(tmp_path, changes) -> None:
    # 配置冲突必须早于任何数据/网络/硬件访问报错。
    config = DeploymentConfig(base_camera=None, dataset_root=str(tmp_path),
                              dataset_repo="local/test", frame_index_prompt_frames=7)

    # 拒绝非法总帧数、空相机名称、重复视角与越界起点；帧prompt本身不限制相机选择。
    with pytest.raises(AssertionError):
        replace(config, **changes).validate()


def test_frame_prompt_robot_never_constructs_camera(tmp_path, monkeypatch) -> None:
    # 真机帧prompt模式可读实测state，但连接工厂不得构造或打开相机。
    (tmp_path / "arm.json").write_text("{}")
    config = DeploymentConfig(observation_source="robot", base_camera=None, wrist_camera=None,
                              frame_index_prompt_frames=7, prompt="Knock down the cylinder",
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


@pytest.mark.parametrize("flags", [{}, {"frame_index_prompt": True}, {"no_vision": True},
                                   {"frame_index_prompt": True, "no_vision": False}])
def test_frame_prompt_rejects_wrong_server(flags) -> None:
    # 本recipe的无图请求不能送给普通vision服务；握手必须明确帧prompt与no-vision两项契约。
    metadata = {key: value for key, value in Policy.metadata.items()
                if key not in ("frame_index_prompt", "no_vision")}
    metadata.update(flags)
    config = DeploymentConfig(frame_index_prompt_frames=7)

    # 新模式提前拒绝缺失/错误标志，普通模式仍只校验原推理结构。
    with pytest.raises(AssertionError, match="frame prompt requires"):
        runner.validate_metadata(metadata, config)
    assert runner.validate_metadata(metadata, DeploymentConfig()) == 10
    assert runner.validate_metadata(Policy.metadata, config) == 10


@pytest.mark.parametrize("frames", [None, 10001])
@pytest.mark.parametrize("base,wrist", [(None, None), ("top", None), (None, "fpv"),
                                       ("top", "fpv")])
def test_prompt_and_camera_config_are_independent(tmp_path, frames, base, wrist) -> None:
    # 帧prompt与视角选择正交：启用或关闭帧条件均可独立选择零、一或两个相机。
    (tmp_path / "arm.json").write_text("{}")
    expected = {key: value for key, value in (("observation/image", base),
                                             ("observation/wrist_image", wrist))
                if value is not None}
    config = DeploymentConfig(
        observation_source="robot", base_camera=base, wrist_camera=wrist,
        cameras={name: {} for name in expected.values()}, frame_index_prompt_frames=frames,
        prompt="Knock down the cylinder", robot_port="fake", robot_id="arm",
        calibration_dir=str(tmp_path),
    )

    # 仅验证纯配置，不调用SDK；10001总帧数必须合法，不残留四位图片编码上限。
    config.validate()
    assert config.image_keys == expected

    # 真机相机配置必须始终精确匹配所选视角，帧prompt不能绕过这一验证。
    with pytest.raises(AssertionError, match="selected camera views"):
        replace(config, cameras={**config.cameras, "unexpected": {}}).validate()
