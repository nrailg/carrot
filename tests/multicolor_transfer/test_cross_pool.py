import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from carrot.distributed import Cluster
from carrot.trainer.rl.config import load_config
from carrot.trainer.rl.trainer import _launch, copy_artifacts


def check_cross_pool(config_path: str, output: Path, simulator_output: str) -> None:
    # 在真实两池验证节点/GPU 放置、图像传输、PPO 闭环与 LIBERO 子进程切换。
    config = replace(
        load_config(config_path),
        output=str(output),
        simulator_output=simulator_output,
        num_envs=32,
        global_batch_size=32,
        global_rollout_batch_size=32,
    )
    assert config.num_nodes == 2 and os.environ.get("RAY_ADDRESS")
    output.mkdir(parents=True, exist_ok=False)
    with Cluster(
        namespace="carrot-transfer-test",
        env_vars={"PYTHONPATH": os.environ["PYTHONPATH"], "HF_HUB_OFFLINE": "1"},
    ) as cluster:
        simulator, learner = _launch(config, cluster)
        print("WORKERS_READY", flush=True)
        # partial reset 不推进其他 slot；真实相机输出经跨池传输后必须完整保留。
        reset_check = simulator.call("request", "check_reset").wait()[0]
        observations = simulator.call("request", "reset", seeds=list(range(7, 39))).wait()[0]
        assert len(observations) == 32 and reset_check["unchanged_slots"] == 31
        for observation in observations:
            assert set(observation) == {
                "observation/image",
                "observation/wrist_image",
                "observation/state",
                "prompt",
            }
            for key in ("image", "wrist_image"):
                frame = observation[f"observation/{key}"]
                assert frame.shape == (224, 224, 3) and frame.dtype == np.uint8
                assert frame.std() > 1, key
            assert observation["observation/state"].shape == (8,)
        simulator.call("request", "snapshot", output=simulator_output + "/snapshot").wait()
        copied = copy_artifacts(simulator, "snapshot", output / "snapshot")
        assert copied["files"] > 0 and copied["bytes"] > 0
        # 使用真实图像执行一个动作块再反传，保护 record 前终态观测和实际步数。
        actions = learner.call("sample", observations).wait()[0]
        result = simulator.call("request", "step", actions=actions[:, :5]).wait()[0]
        counted = learner.call("record", result).wait()[0]
        metrics = learner.call("update").wait()[0]
        assert counted["transitions"] == int(result["lengths"].sum())
        assert metrics["transitions"] == counted["transitions"] > 0
        assert metrics["max_preupdate_logprob_error"] < 0.02
        assert metrics["global_rollout_batch_size"] == 32
        assert metrics["gradient_accumulation_steps"] == 4
        print(json.dumps({"PPO": metrics}), flush=True)
        # 两个仿真依赖环境必须在同一 Pro5000 actor 中顺序切换，H20 只做推理。
        simulator.call("restart", config.libero_script, {}, config.libero_python).wait()
        learner.call("load", config.checkpoint, "libero").wait()
        cases = []
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            first = simulator.call(
                "request", "reset", suite=suite, task_id=0, episode=0, seed=7
            ).wait()[0]
            actions = learner.call("infer", first["observations"], seed=7).wait()[0]
            result = simulator.call("request", "step", actions=actions[:, :5]).wait()[0]
            assert 1 <= result["info"]["controls"] <= 5
            assert np.isfinite(result["observations"][0]["observation/state"]).all()
            # 重置同一 case 的物理初态和像素应一致，避免配对评测产生隐藏条件差异。
            second = simulator.call(
                "request", "reset", suite=suite, task_id=0, episode=0, seed=7
            ).wait()[0]
            assert first["info"] == second["info"]
            for key in ("observation/image", "observation/wrist_image", "observation/state"):
                np.testing.assert_array_equal(
                    first["observations"][0][key], second["observations"][0][key]
                )
            cases.append(result["info"])
            print(json.dumps({"LIBERO": result["info"]}), flush=True)
    report = {
        "status": "PASS",
        "reset": reset_check,
        "transfer": copied,
        "ppo": metrics,
        "libero": cases,
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulator-output", required=True)
    args = parser.parse_args()
    check_cross_pool(args.config, args.output, args.simulator_output)
