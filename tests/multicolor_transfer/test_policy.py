import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from carrot.models.pi05.model import PI0Pytorch
from carrot.trainer.rl.algorithms import chunk_logprob
from carrot.trainer.rl.config import load_config
from carrot.trainer.rl.learner import LearnerWorker


def check_policy(config_path: str, output: Path) -> None:
    # 真 checkpoint 验证概率重算、反传、冻结参数、FP32 恢复和 BF16 导出，禁止用 mock 模型代替。
    config = replace(load_config(config_path), output=str(output), learner_node_ip="127.0.0.1")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(LOCAL_RANK="0", MASTER_ADDR="127.0.0.1")
    learner = LearnerWorker(config)
    learner.setup()
    print("MODEL_LOADED", flush=True)
    policy = learner.policy
    assert policy is not None
    rng = np.random.default_rng(7)
    observations = [
        {
            "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.array([0.4, 0, 0.3, 3.14, 0, 0, 0.04, -0.04], np.float32),
            "prompt": "pick up the red pen from the rack",
        }
        for _ in range(2)
    ]
    before_weight = policy.model.action_out_proj.weight.detach().clone()
    # 用真实不同执行长度核对 ratio=1，终止与截断的下一观测保留给 critic。
    actions = learner.sample(observations)
    assert actions.shape == (2, 10, 7) and np.isfinite(actions).all()
    replay = learner.pending
    assert replay is not None
    with torch.no_grad():
        new, _ = policy.evaluate(observations, replay)
    lengths = torch.tensor([1, 5])
    error = (
        (
            chunk_logprob(new, lengths.to(policy.device)).cpu()
            - chunk_logprob(replay["logprob"], lengths)
        )
        .abs()
        .max()
        .item()
    )
    assert error < 0.02, error
    learner.record(
        {
            "observations": observations,
            "rewards": np.array([1.0, -0.2], np.float32),
            "lengths": lengths.numpy(),
            "terminated": np.array([True, False]),
            "truncated": np.array([False, True]),
        }
    )
    # 更新必须改变动作参数，所有 VLM 参数无梯度且不在优化器参数列表中。
    metrics = learner.update()
    print(json.dumps({"PPO_UPDATE": metrics}), flush=True)
    assert all(np.isfinite(value) for value in metrics.values())
    assert not torch.equal(before_weight, policy.model.action_out_proj.weight)
    frozen = list(policy.model.paligemma_with_expert.paligemma.parameters())
    assert all(not p.requires_grad and p.grad is None for p in frozen)
    assert all(p.dtype == torch.float32 for p in policy.parameters())
    optimized = {id(p) for group in learner.optimizer.param_groups for p in group["params"]}
    assert not optimized.intersection(id(p) for p in frozen)
    # SFT 使用同一输入契约，只有实际子任务范围内的动作进入 loss。
    supervised = [{**obs, "actions": np.zeros((10, 7), np.float32)} for obs in observations]
    learner.optimizer.zero_grad(set_to_none=True)
    loss = policy.supervised_loss(supervised, torch.tensor([1, 3]))
    assert torch.isfinite(loss)
    loss.backward()
    sft_loss = float(loss.detach())
    assert policy.model.action_in_proj.weight.grad is not None
    assert all(p.grad is None for p in frozen)
    learner.optimizer.zero_grad(set_to_none=True)
    # 扰乱权重、RNG 和 SFT 基线来源后，恢复必须同时保留数值状态与跨次续训的对照来源。
    checkpoint = output / "checkpoint"
    learner.save(str(checkpoint))
    print("CHECKPOINT_SAVED", flush=True)
    saved_weight = policy.model.action_out_proj.weight.detach().clone()
    expected_torch, expected_numpy = torch.rand(3), np.random.rand(3)
    expected_cuda = torch.rand(3, device=policy.device)
    with torch.no_grad():
        policy.model.action_out_proj.weight.add_(0.01)
    torch.manual_seed(999)
    np.random.seed(999)
    learner.sft_checkpoint = str(output / "different-run/checkpoints/sft")
    learner.resume(str(checkpoint))
    assert learner.sft_checkpoint == str(output / "checkpoints/sft")
    assert torch.equal(saved_weight, policy.model.action_out_proj.weight)
    assert torch.equal(expected_torch, torch.rand(3))
    assert np.array_equal(expected_numpy, np.random.rand(3))
    assert torch.equal(expected_cuda, torch.rand(3, device=policy.device))
    assert learner.transitions == 6 and learner.updates == 1
    assert all(
        value.dtype == torch.float32
        for state in learner.optimizer.state.values()
        for name, value in state.items()
        if name in ("exp_avg", "exp_avg_sq")
    )
    # 原生导出必须能直接重载，且推理不依赖训练态 critic 或 optimizer。
    del policy, frozen, before_weight, saved_weight, loss, replay
    learner.load(str(checkpoint))
    inference = learner.infer(observations, seed=7)
    assert inference.shape == (2, 10, 7) and np.isfinite(inference).all()
    # 官方 pi05_libero 只启用双相机；接口中的本体状态不能悄悄进入策略。
    encoded = learner.policy.inputs(dict(observations[0]))
    assert encoded["image_mask"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": False,
    }
    changed_state = [
        {**obs, "observation/state": np.arange(8, dtype=np.float32)} for obs in observations
    ]
    np.testing.assert_array_equal(inference, learner.infer(changed_state, seed=7))
    config_json = json.loads((checkpoint / "config.json").read_text())
    assert config_json["action_horizon"] == 10
    native = PI0Pytorch.load_config(checkpoint)
    assert native["action_dim"] >= 7
    report = {
        "status": "PASS",
        "camera_inputs": 2,
        "state_invariance": "PASS",
        "preupdate_logprob_error": error,
        "ppo": metrics,
        "sft_loss": sft_loss,
        "checkpoint": str(checkpoint),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_policy(args.config, args.output)
