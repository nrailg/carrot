import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from carrot.models.pi05.inference import create_libero_policy
from carrot.trainer.rl.config import load_config
from carrot.trainer.rl.data import Demonstrations
from carrot.trainer.rl.learner import LearnerWorker


def select_samples(dataset: Demonstrations) -> tuple[list[dict], list[dict]]:
    # 各课程选两条成功示范，每个子任务取两个内部时刻，避免只检查容易的静止帧。
    chosen: dict[int, int] = {}
    samples, metadata = [], []
    base = 0
    for episode_id, episode in enumerate(dataset.episodes):
        phases = np.unique(episode["phase"])
        colors = (int(phases.max()) + 2) // 3
        if chosen.get(colors, 0) < 2:
            chosen[colors] = chosen.get(colors, 0) + 1
            for phase in phases:
                if phase == colors * 3:
                    continue
                offsets = np.flatnonzero(episode["phase"] == phase)
                for fraction in (1 / 3, 2 / 3):
                    offset = int(offsets[int((len(offsets) - 1) * fraction)])
                    sample, length = dataset[base + offset]
                    samples.append(sample)
                    metadata.append(
                        {
                            "episode": episode_id,
                            "offset": offset,
                            "phase": int(phase),
                            "colors": colors,
                            "length": length,
                        }
                    )
        base += len(episode["phase"])
    assert chosen == {1: 2, 2: 2, 3: 2}, chosen
    return samples, metadata


def metrics(predictions: np.ndarray, targets: np.ndarray, metadata: list[dict]) -> dict:
    # 仅比较实际会执行且未跨子任务的前五步；分维度统计，避免夹爪掩盖位移误差。
    mask = np.arange(5)[None, :] < np.array([m["length"] for m in metadata])[:, None]
    error = np.abs(predictions[:, :5] - targets[:, :5])
    valid = error[mask]
    return {
        "mae_per_dimension": valid.mean(axis=0).tolist(),
        "p95_per_dimension": np.quantile(valid, 0.95, axis=0).tolist(),
        "gripper_sign_accuracy": float(
            (np.sign(predictions[:, :5, 6])[mask] == np.sign(targets[:, :5, 6])[mask]).mean()
        ),
        "first_action_mae_per_sample": error[:, 0].mean(axis=1).tolist(),
        "out_of_bounds_fraction": float((np.abs(predictions[:, :5][mask]) > 1).mean()),
    }


def check_inference(config_path: str, output: Path, steps: list[int]) -> None:
    # 真示范、真权重检查输入输出可逆及两条推理路径；不以离线误差替代闭环成功率。
    config = replace(load_config(config_path), learner_node_ip="127.0.0.1")
    assert config.learner_demonstrations is not None
    dataset = Demonstrations(config.learner_demonstrations)
    raw, metadata = select_samples(dataset)
    targets = np.stack([r["actions"] for r in raw])
    output.mkdir(parents=True, exist_ok=False)
    (output / "samples.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.save(output / "targets.npy", targets)
    os.environ.update(LOCAL_RANK="0", MASTER_ADDR="127.0.0.1")
    config = replace(
        config,
        checkpoint=str(Path(config.output) / "checkpoints" / f"sft-step-{steps[0]:06d}"),
    )
    learner = LearnerWorker(config)
    learner.setup()
    results = []
    for step in steps:
        # 每个checkpoint用相同示范、噪声seed和micro batch；不修改正式训练的进程或文件。
        checkpoint = Path(config.output) / "checkpoints" / f"sft-step-{step:06d}"
        if step != steps[0]:
            learner.load(str(checkpoint))
        policy = learner.policy
        assert policy is not None
        values: dict[str, list[np.ndarray]] = {"ode10": [], "sde5": [], "native10": []}
        losses, roundtrip = [], []
        with torch.no_grad():
            for start in range(0, len(raw), config.micro_batch_size):
                end = start + config.micro_batch_size
                batch = raw[start:end]
                observation, transformed = policy.encode(batch)
                decoded = (
                    policy.outputs(
                        {
                            "state": observation.state,
                            "actions": transformed["actions"].to(policy.device),
                        }
                    )["actions"]
                    .cpu()
                    .numpy()
                )
                roundtrip.append(float(np.abs(decoded - targets[start:end]).max()))
                lengths = torch.tensor([m["length"] for m in metadata[start:end]])
                torch.manual_seed(7000 + start)
                losses.append(float(policy.supervised_loss(batch, lengths)))
                for name, stochastic in (("ode10", False), ("sde5", True)):
                    torch.manual_seed(9000 + start)
                    values[name].append(policy.sample(batch, stochastic=stochastic)["actions"])
                torch.manual_seed(9000 + start)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latent = policy.model.sample_actions(policy.device, observation, num_steps=10)
                values["native10"].append(
                    policy.outputs({"state": observation.state, "actions": latent})["actions"]
                    .float()
                    .cpu()
                    .numpy()
                )
        # 持久化完整预测以便定位具体帧；只断言数值和变换契约，效果差异如实记录。
        predicted = {name: np.concatenate(parts) for name, parts in values.items()}
        assert all(np.isfinite(a).all() for a in predicted.values())
        assert max(roundtrip) < 1e-5, roundtrip
        for name, value in predicted.items():
            np.save(output / f"step-{step:06d}-{name}.npy", value)
        # 检查真实部署精度路径，避免只在训练侧FP32权重上比较两个ODE实现。
        deployed = create_libero_policy(
            checkpoint,
            device=str(policy.device),
            tokenizer_path=config.tokenizer,
            norm_stats_path=config.norm_stats,
        )
        deployed_actions, learner_actions = [], []
        for sample_id, sample in enumerate(raw[:8]):
            torch.manual_seed(11000 + sample_id)
            noise = torch.randn(1, 10, policy.model.config.action_dim, device=policy.device)
            deployed_actions.append(deployed.infer(sample, noise=noise[0].cpu().numpy())["actions"])
            observation, _ = policy.encode([sample])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent = policy.model.sample_actions(policy.device, observation, noise=noise)
            learner_actions.append(
                policy.outputs({"state": observation.state, "actions": latent})["actions"][0]
                .float()
                .cpu()
                .numpy()
            )
        del deployed
        row = {
            "step": step,
            "samples": len(raw),
            "flow_loss": float(np.mean(losses)),
            "roundtrip_max_error": max(roundtrip),
            "native_ode_max_difference": float(
                np.abs(predicted["ode10"] - predicted["native10"]).max()
            ),
            "deployed_learner_max_difference": float(
                np.abs(np.stack(deployed_actions) - np.stack(learner_actions)).max()
            ),
            "modes": {name: metrics(value, targets, metadata) for name, value in predicted.items()},
        }
        results.append(row)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        compact = {
            **row,
            "modes": {
                name: {k: v for k, v in value.items() if k != "first_action_mae_per_sample"}
                for name, value in row["modes"].items()
            },
        }
        print(json.dumps(compact), flush=True)
        assert row["native_ode_max_difference"] < 1e-5, row["native_ode_max_difference"]
    (output / "completed.json").write_text(
        json.dumps({"steps": steps, "closed_loop": False}) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", nargs="+", type=int, default=[50, 100, 250, 500])
    args = parser.parse_args()
    check_inference(args.config, args.output, args.steps)
