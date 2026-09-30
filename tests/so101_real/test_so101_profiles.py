from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from examples.so101_real import dataset as dataset_module
from examples.so101_real import runner
from examples.so101_real.actions import LogSink, SO101Sink
from examples.so101_real.config import JOINT_NAMES, DeploymentConfig, load_config
from examples.so101_real.dataset import load_dataset_source
from examples.so101_real.observations import DatasetSource, RobotSource
from examples.so101_real.runner import RunLog, _action_limits, run_loop
from tests.so101_real.test_so101_runtime import Dataset, Policy, Robot


class WristDataset(Dataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["observation.images.wrist"] = sample.pop("observation.images.fpv")
        del sample["observation.images.top"]
        return sample


@pytest.mark.parametrize("fps", [0, -1, True, 1.5])
def test_invalid_fps_rejected(tmp_path, fps):
    # 非正数、布尔值或非整数不能作为动作节拍；配置阶段就应拒绝。
    config = DeploymentConfig(dataset_root=str(tmp_path), fps=fps)

    # 校验发生在资源连接前，避免错误帧率进入执行循环。
    with pytest.raises(AssertionError, match="FPS"):
        config.validate()


def test_wipe_profile_and_camera_validation(tmp_path):
    # wipe模板必须显式使用单腕、15FPS和度数；自采本地数据允许没有Hub revision。
    config = load_config(
        Path("examples/so101_real/wipe.yaml"), {"dataset_root": str(tmp_path)},
    )

    # 新配置仍默认只记日志，并且不伪造不存在的base视角。
    assert config.fps == 15 and config.use_degrees is True
    assert config.dataset_revision is None
    assert config.image_keys == {"observation/wrist_image": "wrist"}
    assert config.observation_source == "dataset" and config.action_sink == "log"
    assert config.execute_steps == config.max_chunks == 1

    # 重复映射同一相机不能冒充两个视角。
    config.base_camera = "wrist"
    with pytest.raises(AssertionError, match="distinct"):
        config.validate()


def test_single_wrist_requests_padding_and_logs(tmp_path, monkeypatch):
    # 单腕episode仍按原边界截断，缺失base视角不发送、不复制，也不保存伪造图像。
    config = DeploymentConfig(base_camera=None, wrist_camera="wrist", fps=15, use_degrees=True,
                              execute_steps=3, max_chunks=None)
    source = DatasetSource(WristDataset(), 2, 0, 4, image_keys=config.image_keys)
    policy = Policy()
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    # 执行真实循环和日志，预热及三个动作块共享同一单腕输入契约。
    with ExitStack() as stack:
        log = RunLog(tmp_path)
        stack.callback(log.close)
        result = run_loop(config, source, LogSink(), policy, log)

    # 末块仅一帧有效，模型请求中始终没有示教动作和base图像。
    assert result == {"reason": "episode_end", "chunks": 3, "steps": 7}
    assert all(set(request) == {"observation/state", "observation/wrist_image", "prompt"}
               for request in policy.requests)
    assert np.all(policy.requests[0]["observation/wrist_image"] == 45)
    assert (tmp_path / "first_wrist.png").is_file()
    assert not (tmp_path / "first_top.png").exists()
    with np.load(tmp_path / "chunk_000002.npz") as chunk:
        assert chunk["reference"].shape == (1, 6)


def test_robot_single_wrist_source():
    # 实时源按同一映射读取腕图，不要求top/fpv；关节状态保持原始度数。
    robot = Mock()
    state = [-12, -105, 96, 54, 150, 0.5]
    robot.get_observation.return_value = {
        **dict(zip(JOINT_NAMES, state, strict=True)),
        "wrist": np.full((12, 16, 3), 17, dtype=np.uint8),
    }

    # 单腕请求只从一次真实观测组装，不做关节单位转换或视角复制。
    frame = RobotSource(robot, "Move an object", 50,
                        image_keys={"observation/wrist_image": "wrist"}).read()

    # 超过100度的关节不能被当成归一化位置截断。
    np.testing.assert_array_equal(frame.request["observation/state"], np.float32(state))
    assert "observation/image" not in frame.request
    assert frame.request["observation/wrist_image"].shape == (12, 16, 3)
    robot.get_observation.assert_called_once()


def test_15fps_pacing_accounts_for_send_time(tmp_path, monkeypatch):
    # 15FPS每步预算1/15秒，并扣除发送耗时；不能继续使用硬编码30Hz。
    config = DeploymentConfig(fps=15, execute_steps=2)
    ticks = iter([0, 0.01, 1, 1.01, 2, 2.02])
    sleeps = []
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)

    # 固定耗时模拟两次发送，避免依赖机器速度而产生时间测试抖动。
    with ExitStack() as stack:
        log = RunLog(tmp_path)
        stack.callback(log.close)
        run_loop(config, DatasetSource(Dataset(), 2, 0, 4), LogSink(), Policy(), log)

    # 两步分别消耗10ms和20ms，等待后仍各自满足15FPS预算。
    assert sleeps == pytest.approx([1 / 15 - 0.01, 1 / 15 - 0.02])


@pytest.mark.parametrize("bad_metadata", [None, "fps", "camera", "joints", "robot_type"])
def test_wipe_dataset_metadata_and_time_window(tmp_path, monkeypatch, bad_metadata):
    # 自采so_follower别名和单腕15FPS元信息合法；缺视角、错误关节或频率必须提前失败。
    (tmp_path / "meta").mkdir()
    for file in ("info.json", "stats.json", "tasks.parquet"):
        (tmp_path / "meta" / file).touch()
    config = DeploymentConfig(dataset_root=str(tmp_path), dataset_revision=None,
                              dataset_repo="local/wipe", base_camera=None,
                              wrist_camera="wrist", fps=15, use_degrees=True, episode=2)
    features = {name: {"shape": [6], "names": list(JOINT_NAMES)}
                for name in ("observation.state", "action")}
    features["observation.images.wrist"] = {"dtype": "video", "shape": [480, 640, 3]}
    meta = SimpleNamespace(robot_type="so_follower", fps=15, features=features)
    if bad_metadata == "fps":
        meta.fps = 30
    elif bad_metadata == "camera":
        del features["observation.images.wrist"]
    elif bad_metadata == "joints":
        features["action"]["names"] = list(reversed(JOINT_NAMES))
    elif bad_metadata == "robot_type":
        meta.robot_type = "another_robot"
    factory = Mock(return_value=WristDataset())
    monkeypatch.setattr(dataset_module, "LeRobotDatasetMetadata", lambda *args, **kwargs: meta)
    monkeypatch.setattr(dataset_module, "LeRobotDataset", factory)

    # 错误metadata不得进入数据构造；合法数据保留原FPS对应的动作时间窗口。
    if bad_metadata is not None:
        with pytest.raises(AssertionError):
            load_dataset_source(config, 4)
        factory.assert_not_called()
    else:
        source = load_dataset_source(config, 4)
        assert factory.call_args.kwargs["delta_timestamps"] == {
            "action": [0, 1 / 15, 2 / 15, 3 / 15],
        }
        assert factory.call_args.kwargs["revision"] is None
        assert "observation/image" not in source.read().request


@pytest.mark.parametrize("use_degrees", [False, True])
def test_calibrated_action_limits_and_raw_targets(use_degrees):
    # 用真实LeRobot总线纯换算作为限位oracle；构造总线不会打开串口或写寄存器。
    robot = Robot()
    names = [name.removesuffix(".pos") for name in JOINT_NAMES]
    robot.bus = FeetechMotorsBus(
        port="fake",
        motors={name: Motor(index, "sts3215", MotorNormMode.RANGE_0_100 if name == "gripper"
                            else MotorNormMode.DEGREES if use_degrees
                            else MotorNormMode.RANGE_M100_100)
                for index, name in enumerate(names, 1)},
        calibration={name: MotorCalibration(
            index, 0, 0, 0 if name == "wrist_roll" else 1000,
            {"wrist_roll": 4095, "elbow_flex": 3213, "wrist_flex": 3198}.get(name, 3000),
        )
                     for index, name in enumerate(names, 1)},
    )
    lower, upper = _action_limits(robot)
    sink = SO101Sink(robot, 10, lower, upper)
    target = np.array([-300, 300, -300, -300, 150, -10], dtype=np.float32)

    # 发送前的绝对裁剪应让所有raw目标落在标定范围内，夹爪始终使用0..100百分比。
    result = sink.send(target)
    raw = robot.bus._unnormalize({robot.bus.motors[name].id: robot.commands[0][f"{name}.pos"]
                                 for name in names})

    # 度数模式允许腕旋转150度；旧归一化模式仍限制到100，检测错误共用限位的回归。
    assert result["bounded_target"][4] == (150 if use_degrees else 100)
    assert result["bounded_target"][5] == 0
    for name in names:
        calibration = robot.bus.calibration[name]
        assert calibration.range_min <= raw[robot.bus.motors[name].id] <= calibration.range_max
    assert not robot.bus.is_connected

    # 实际位置越过标定范围时，不允许相对当前位置限幅生成仍越界的目标。
    count = len(robot.commands)
    robot.state[0] = upper[0] + 1
    with pytest.raises(AssertionError, match="outside calibrated"):
        sink.send(target)
    assert len(robot.commands) == count


def test_connect_robot_uses_degree_profile_and_validates_camera(tmp_path, monkeypatch):
    # 真机工厂应透传度数和单腕配置；用替身保证测试绝不打开真实设备。
    config = load_config(Path("examples/so101_real/wipe.yaml"),
                         {"dataset_root": str(tmp_path)})
    config.observation_source = "robot"
    config.robot_port, config.robot_id, config.calibration_dir = "fake", "arm", str(tmp_path)
    robot = Mock()
    robot.calibration = {"fixture": True}
    robot.is_calibrated = True
    robot.bus.is_connected = False
    robot.cameras = {}
    factory = Mock(return_value=robot)
    monkeypatch.setattr(runner, "SO101Follower", factory)

    # 检查正常连接参数，禁用自动校准与断开时关闭扭矩的行为保持原契约。
    with ExitStack() as stack:
        runner.connect_robot(config, stack)
        actual = factory.call_args.args[0]
        assert actual.use_degrees is True and set(actual.cameras) == {"wrist"}
        assert actual.cameras["wrist"].fps == 15
        assert actual.disable_torque_on_disconnect is False
        robot.connect.assert_called_once_with(calibrate=False)

    # 摄像头FPS不匹配时必须在构造机器人前拒绝，而非默默改配置。
    factory.reset_mock()
    config.cameras["wrist"]["fps"] = 30
    with ExitStack() as stack, pytest.raises(AssertionError, match="camera FPS"):
        runner.connect_robot(config, stack)
    factory.assert_not_called()
