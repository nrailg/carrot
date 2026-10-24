import argparse
import json
import os
import time
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
    config = replace(
        load_config(config_path),
        output=str(output),
        learner_node_ip="127.0.0.1",
        global_batch_size=256,
        global_rollout_batch_size=256,
        micro_batch_size=8,
        sft_steps=1,
        sft_checkpoint_every_steps=1,
    )
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
        for _ in range(32)
    ]
    before_weight = policy.model.action_out_proj.weight.detach().clone()
    # 用真实不同执行长度核对 ratio=1，终止与截断的下一观测保留给 critic。
    actions = learner.sample(observations)
    assert actions.shape == (32, 10, 7) and np.isfinite(actions).all()
    replay = learner.pending
    assert replay is not None
    with torch.no_grad():
        new, _ = policy.evaluate(
            observations[:8], {k: v[:8] for k, v in replay.items() if isinstance(v, torch.Tensor)}
        )
    lengths = torch.tensor([1, 5] * 16)
    error = (
        (
            chunk_logprob(new, lengths[:8].to(policy.device)).cpu()
            - chunk_logprob(replay["logprob"][:8], lengths[:8])
        )
        .abs()
        .max()
        .item()
    )
    assert error < 0.02, error
    result = {
        "observations": observations,
        "rewards": np.array([1.0, -0.2] * 16, np.float32),
        "lengths": lengths.numpy(),
        "terminated": np.array([True, False] * 16),
        "truncated": np.array([False, True] * 16),
    }
    learner.record(result)
    # 正式 GBS=256 需累积 32 次 micro=8；不足一个 rollout 时必须拒绝更新。
    try:
        learner.update()
    except AssertionError as exc:
        assert "256 rollout samples" in str(exc)
    else:
        raise AssertionError("partial rollout was accepted")
    for _ in range(7):
        learner.sample(observations)
        learner.record(result)
    # 更新必须改变动作参数，所有 VLM 参数无梯度且不在优化器参数列表中。
    metrics = learner.update()
    print(json.dumps({"PPO_UPDATE": metrics}), flush=True)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["global_batch_size"] == 256 and metrics["optimizer_steps"] == 1
    assert metrics["gradient_accumulation_steps"] == 32
    assert not torch.equal(before_weight, policy.model.action_out_proj.weight)
    frozen = list(policy.model.paligemma_with_expert.paligemma.parameters())
    assert all(not p.requires_grad and p.grad is None for p in frozen)
    assert all(p.dtype == torch.float32 for p in policy.parameters())
    optimized = {id(p) for group in learner.optimizer.param_groups for p in group["params"]}
    assert not optimized.intersection(id(p) for p in frozen)
    # SFT 使用同一输入契约，只有实际子任务范围内的动作进入 loss。
    supervised = [{**obs, "actions": np.zeros((10, 7), np.float32)} for obs in observations[:2]]
    learner.optimizer.zero_grad(set_to_none=True)
    # 真 SFT 入口完成一次 GBS=256 更新，覆盖采样、梯度累积和审计日志。
    episode = output / "demos/episode-0000"
    episode.mkdir(parents=True)
    for key in observations[0]:
        np.save(
            episode / (key.replace("observation/", "") + ".npy"),
            np.asarray([obs[key] for obs in observations]),
        )
    np.save(episode / "actions.npy", np.zeros((32, 7), np.float32))
    np.save(episode / "phase.npy", np.repeat(np.arange(8), 4))
    (episode / "result.json").write_text(json.dumps({"success": True}))
    sft_result = learner.sft(str(output / "demos"), time.time() + 1800)
    assert sft_result["sft_updates"] == 1
    sft_row = json.loads((output / "sft.jsonl").read_text())
    assert sft_row["global_batch_size"] == 256 and sft_row["samples_seen"] == 256
    # 每步保存用于加速覆盖正式50步保存路径，检查原子完成及真实SFT优化器状态。
    periodic = output / "checkpoints/sft-step-000001"
    assert sft_row["checkpoint"] == str(periodic)
    assert (periodic / "model.safetensors").is_file()
    assert not periodic.with_name(periodic.name + ".partial").exists()
    periodic_manifest = json.loads((periodic / "manifest.json").read_text())
    assert periodic_manifest["stage"] == "sft" and periodic_manifest["sft_updates"] == 1
    saved_sft = torch.load(periodic / "learner.pt", map_location="cpu", weights_only=False)
    assert saved_sft["sft_optimizer"]["state"]
    assert all(s["step"] == 1 for s in saved_sft["sft_optimizer"]["state"].values())
    # 保存的FP32训练参数必须是本次更新之后的权重。
    torch.testing.assert_close(
        saved_sft["trainable"]["action_out_proj.weight"],
        policy.model.action_out_proj.weight.detach().cpu(),
        rtol=0,
        atol=0,
    )
    del saved_sft
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
    assert learner.transitions == 768 and learner.updates == 1 and learner.sft_updates == 1
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
    assert inference.shape == (32, 10, 7) and np.isfinite(inference).all()
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
    # 截止时尾部只丢弃，不形成小 batch，也不增加 optimizer/update 计数。
    learner.sample(observations)
    learner.record(result)
    discarded = learner.discard_rollout()
    assert discarded == {"discarded_samples": 32, "updates": 1}
    assert not learner.buffer
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
        "sft_batch": sft_row,
        "discarded_tail": discarded,
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
