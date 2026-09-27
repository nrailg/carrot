import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

from examples.so101_real import runner
from examples.so101_real.config import DeploymentConfig


def test_dataset_through_real_policy_server(monkeypatch) -> None:
    # 真数据经客户端和独立 GPU 服务进程返回动作，日志模式不得触碰机器人。
    directory = Path(os.environ["CARROT_SO101_WEBSOCKET_RUN"])
    directory.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def forbid_robot(*args, **kwargs):
        raise AssertionError("dataset+log must not connect to a robot")

    monkeypatch.setattr(runner, "connect_robot", forbid_robot)
    config = DeploymentConfig(
        dataset_root=os.environ["CARROT_SO101_DATASET"],
        server_uri=f"ws://127.0.0.1:{port}",
        output_dir=str(directory / "client"),
        execute_steps=2, max_chunks=2,
        warmup_timeout_s=120, request_timeout_s=120,
    )

    # 使用正式 server CLI 加载真实 checkpoint；健康检查只重试服务启动期连接失败。
    command = [
        sys.executable, "-m", "carrot.cli.serve_pi05_policy",
        "--embodiment", "so101", "--checkpoint", os.environ["CARROT_SO101_CHECKPOINT"],
        "--device", "cuda:0", "--host", "127.0.0.1", "--port", str(port),
    ]
    (directory / "server_command.json").write_text(json.dumps(command))
    with (directory / "server.log").open("w") as server_log:
        process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                assert process.poll() is None, f"server exited; see {directory / 'server.log'}"
                assert time.monotonic() < deadline, "policy server startup exceeded 180 seconds"
                try:
                    with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                        assert response.status == 200 and response.read() == b"OK\n"
                    break
                except URLError:
                    time.sleep(0.2)

            # 执行正式客户端组装入口：真实读数据、网络推理、两轮动作日志及图表生成。
            runner.run(config)
            assert process.poll() is None, "server exited during client session"
        finally:
            # 只结束本测试创建的服务进程，成功或失败都必须回收 GPU 资源。
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)

    # 两个 chunk 消费四帧；动作只写日志，保存的预测必须来自有限的 float32[50,6]。
    client = directory / "client"
    summary = json.loads((client / "summary.json").read_text())
    assert summary == {"reason": "max_chunks", "chunks": 2, "steps": 4, "completed": True}
    events = [json.loads(line) for line in (client / "events.jsonl").read_text().splitlines()]
    metadata = next(event["metadata"] for event in events if event["event"] == "metadata")
    assert metadata["embodiment"] == "so101" and metadata["action_horizon"] == 50
    assert metadata["action_dim"] == 6
    requests = [event for event in events if event["event"] == "request"]
    assert [event["frame"] for event in requests] == [0, 2]
    assert all(event["episode"] == 0 and event["prompt"] for event in requests)
    actions = [event for event in events if event["event"] == "action"]
    assert len(actions) == 4 and all(event["executed"] is False for event in actions)
    for index in range(2):
        with np.load(client / f"chunk_{index:06d}.npz") as chunk:
            prediction = chunk["predicted"]
            assert prediction.shape == (50, 6) and prediction.dtype == np.float32
            assert np.isfinite(prediction).all()
            assert int(chunk["frame"]) == index * 2 and int(chunk["planned_steps"]) == 2
            np.testing.assert_array_equal(chunk["state"], requests[index]["state"])
            np.testing.assert_array_equal(
                prediction[:2], [event["target"] for event in actions if event["chunk"] == index]
            )

    # 报告仅比较实际消费的四帧；MAE 记录调试信息，不作为真机任务效果判据。
    comparison = json.loads((client / "comparison.json").read_text())
    assert comparison["compared_frames"] == 4
    assert np.isfinite(list(comparison["joint_mae"].values())).all()
    for name in ("actions.png", "first_top.png", "first_fpv.png"):
        assert (client / name).stat().st_size > 0
