import json
from unittest.mock import Mock

import numpy as np
import pytest

from examples.so101_real import runner
from examples.so101_real.config import JOINT_NAMES, DeploymentConfig
from examples.so101_real.observations import ObservationFrame


@pytest.mark.parametrize("source,sink", [
    ("dataset", "log"), ("dataset", "robot"), ("robot", "log"), ("robot", "robot"),
])
@pytest.mark.parametrize("fail_request", [False, True])
def test_session_hardware_selection_and_cleanup(tmp_path, monkeypatch, source, sink, fail_request):
    # 验证四种模式真正的组装入口；默认模式不得连接硬件，失败也必须关闭资源并记录原因。
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text("{}")
    (tmp_path / "arm.json").write_text("{}")
    config = DeploymentConfig(
        observation_source=source, action_sink=sink, dataset_root=str(tmp_path),
        output_dir=str(tmp_path / "run"), robot_port="fake", robot_id="arm",
        calibration_dir=str(tmp_path), cameras={"top": {}, "fpv": {}}, prompt="pick",
    )
    images = {"top": np.zeros((12, 16, 3), dtype=np.uint8),
              "fpv": np.zeros((16, 12, 3), dtype=np.uint8)}
    joints = dict.fromkeys(JOINT_NAMES, 0.0)
    robot = Mock()
    robot.get_observation.return_value = {**joints, **images}
    robot.send_action.return_value = joints
    policy = Mock()
    policy.metadata = {"embodiment": "so101", "action_dim": 6,
                       "action_horizon": 4, "num_steps": 10}
    response = {"actions": np.zeros((4, 6), dtype=np.float32)}
    policy.infer.side_effect = [response, TimeoutError("injected") if fail_request else response]

    # 用显式硬件工厂替身替换串口操作，其余配置验证、日志、调度和清理均执行真实代码。
    connections = []

    def connect_robot(cfg, stack):
        connections.append(cfg)
        stack.callback(robot.disconnect)
        return robot

    frame = ObservationFrame(
        request={"observation/state": np.zeros(6, dtype=np.float32),
                 "observation/image": images["top"],
                 "observation/wrist_image": images["fpv"], "prompt": "pick"},
        episode=0, frame=0, reference=np.zeros((4, 6)), valid_steps=4,
    )
    recorded = Mock()
    recorded.read.return_value = frame
    monkeypatch.setattr(runner, "PolicyClient", lambda *args: policy)
    monkeypatch.setattr(runner, "connect_robot", connect_robot)
    monkeypatch.setattr(runner, "load_dataset_source", lambda *args: recorded)
    monkeypatch.setattr(runner, "write_report", lambda directory: None)
    monkeypatch.setattr("examples.so101_real.runner.time.sleep", lambda delay: None)

    # 正常运行应有完整 summary；请求失败仍要释放资源，且无任何动作下发。
    if fail_request:
        with pytest.raises(TimeoutError, match="injected"):
            runner.run(config)
    else:
        runner.run(config)

    # 只有显式选择机器人观测或执行才可连接；日志执行端永远不发动作。
    needs_robot = source == "robot" or sink == "robot"
    assert len(connections) == int(needs_robot)
    assert robot.disconnect.call_count == int(needs_robot)
    assert robot.send_action.call_count == int(sink == "robot" and not fail_request)
    policy.close.assert_called_once()
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["completed"] is not fail_request
    assert summary["reason"] == ("TimeoutError" if fail_request else "max_chunks")
