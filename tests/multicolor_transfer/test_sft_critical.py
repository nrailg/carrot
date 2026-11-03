import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from carrot.models.pi05.model.pi0_pytorch import make_att_2d_masks
from carrot.trainer.rl.algorithms import supervised_flow_loss
from carrot.trainer.rl.config import load_config
from carrot.trainer.rl.data import Demonstrations
from carrot.trainer.rl.learner import LearnerWorker
from carrot.trainer.rl.pi05 import FlowPolicy


def check_forward_contract(
    policy: FlowPolicy, raw: list[dict], lengths: torch.Tensor, output: Path
) -> None:
    # 同一输入和噪声对照联合训练前向与KV缓存前向；速度与梯度都一致才排除训练/推理分叉。
    observation, data = policy.encode(raw)
    targets = data["actions"].to(policy.device)
    torch.manual_seed(7100)
    noise = torch.randn_like(targets)
    time = torch.full((len(raw),), 0.5, device=policy.device)
    latent = time[:, None, None] * noise + (1 - time[:, None, None]) * targets
    model = policy.model
    selected = {
        "action_in": model.action_in_proj.weight,
        "action_out": model.action_out_proj.weight,
        "expert_query": (
            model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight
        ),
    }
    # 缓存路径使用生产实现，联合路径直接走原模型的双塔attention，不引入图像增强差异。
    cached = policy.velocity(policy.prefix(observation), latent, time)
    cached_loss = supervised_flow_loss((cached - (noise - targets)).square(), lengths)
    cached_grad = torch.autograd.grad(cached_loss, list(selected.values()))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        images, masks, tokens, token_masks, state = model._preprocess_observation(
            observation, train=False
        )
        prefix, prefix_pad, prefix_att = model.embed_prefix(images, masks, tokens, token_masks)
        suffix, suffix_pad, suffix_att, conditioning = model.embed_suffix(state, latent, time)
        pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        attention = torch.cat([prefix_att, suffix_att], dim=1)
        (_, hidden), _ = model.paligemma_with_expert.forward(
            attention_mask=model._prepare_attention_masks_4d(make_att_2d_masks(pad, attention)),
            position_ids=pad.cumsum(1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix.to(torch.bfloat16), suffix.to(torch.bfloat16)],
            use_cache=False,
            adarms_cond=[None, conditioning],
        )
        joint = model.action_out_proj(hidden[:, -model.config.action_horizon :].float()).float()
    joint_loss = supervised_flow_loss((joint - (noise - targets)).square(), lengths)
    joint_grad = torch.autograd.grad(joint_loss, list(selected.values()))
    # 保存数值差与梯度方向；BF16舍入容差不允许掩盖显著的前向或反向差异。
    gradients = {}
    for name, a, b in zip(selected, cached_grad, joint_grad, strict=True):
        gradients[name] = {
            "cosine": float(
                torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
            ),
            "relative_l2": float((a - b).norm() / b.norm()),
        }
    result = {
        "velocity_max_error": float((cached.detach() - joint.detach()).abs().max()),
        "velocity_relative_l2": float(
            (cached.detach() - joint.detach()).norm() / joint.detach().norm()
        ),
        "cached_loss": float(cached_loss.detach()),
        "joint_loss": float(joint_loss.detach()),
        "gradients": gradients,
    }
    (output / "forward-contract.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"forward_contract": result}), flush=True)
    assert result["velocity_relative_l2"] < 0.02, result
    assert all(v["cosine"] > 0.99 for v in gradients.values()), result


def check_critical_frames(config_path: str, checkpoint: str, output: Path, fit_steps: int) -> None:
    # 对真实FP32 SFT权重检查关键动作帧及未监督维度，不能用等待帧平均误差掩盖转折错误。
    config = replace(load_config(config_path), learner_node_ip="127.0.0.1")
    assert config.learner_demonstrations is not None
    dataset = Demonstrations(config.learner_demonstrations)
    offsets = [0, 20, 40, 240, 260, 280, 500, 540, 600]
    raw, metadata = [], []
    base = 0
    for episode in range(2):
        for offset in offsets:
            sample, length = dataset[base + offset]
            raw.append(sample)
            metadata.append({"episode": episode, "offset": offset, "length": length})
        base += len(dataset.episodes[episode]["actions"])
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(LOCAL_RANK="0", MASTER_ADDR="127.0.0.1")
    learner = LearnerWorker(config)
    learner.setup()
    learner.resume(checkpoint)
    policy = learner.policy
    assert policy is not None
    check_forward_contract(
        policy,
        [raw[0], raw[6]],
        torch.tensor([metadata[0]["length"], metadata[6]["length"]]),
        output,
    )
    rows = []
    with torch.no_grad():
        for start in range(0, len(raw), config.micro_batch_size):
            # 同一观测/噪声比较正常ODE与强制补齐维度保持训练分布的反事实采样。
            batch = raw[start : start + config.micro_batch_size]
            observation, data = policy.encode(batch)
            prefix = policy.prefix(observation)
            targets = data["actions"].to(policy.device)
            assert torch.count_nonzero(targets[:, :, 7:]) == 0
            torch.manual_seed(12000 + start)
            noise = torch.randn_like(targets)
            predictions, latent_padding = {}, {}
            for name, projected in (("normal", False), ("projected_padding", True)):
                latent = noise.clone()
                time = torch.tensor(1.0, device=policy.device)
                delta = torch.tensor(-0.1, device=policy.device)
                for _ in range(10):
                    if projected:
                        latent[:, :, 7:] = time * noise[:, :, 7:]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        velocity = policy.model.denoise_step(
                            prefix.state, prefix.mask, prefix.cache, latent, time.expand(len(batch))
                        )
                    latent = latent + delta * velocity
                    time = time + delta
                predictions[name] = (
                    policy.outputs({"state": observation.state, "actions": latent})["actions"]
                    .cpu()
                    .numpy()
                )
                latent_padding[name] = latent[:, :, 7:].abs().mean((1, 2)).cpu().tolist()
            # 在实际训练分布上比较有效控制维度与补齐维度的速度误差，定位未约束支路。
            errors = {}
            for time_value in (0.1, 0.5, 0.9):
                time = torch.full((len(batch),), time_value, device=policy.device)
                latent = time[:, None, None] * noise + (1 - time[:, None, None]) * targets
                velocity = policy.velocity(prefix, latent, time)
                squared = (velocity - (noise - targets)).square()
                errors[str(time_value)] = {
                    "control_mse": squared[:, :, :7].mean((1, 2)).cpu().tolist(),
                    "padding_mse": squared[:, :, 7:].mean((1, 2)).cpu().tolist(),
                }
            for i, sample in enumerate(batch):
                row = {
                    **metadata[start + i],
                    "target": sample["actions"][0].tolist(),
                    "predictions": {k: v[i, 0].tolist() for k, v in predictions.items()},
                    "padding_abs_mean": {k: v[i] for k, v in latent_padding.items()},
                    "velocity_error": {
                        t: {k: v[i] for k, v in values.items()} for t, values in errors.items()
                    },
                }
                rows.append(row)
                assert all(np.isfinite(a).all() for a in row["predictions"].values())
            (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    # 不把反事实变好当成修复完成；保留逐帧证据供后续闭环验收。
    summary = {}
    for mode in ("normal", "projected_padding"):
        error = np.array([r["predictions"][mode] for r in rows]) - np.array(
            [r["target"] for r in rows]
        )
        summary[mode] = {"mae": np.abs(error).mean(0).tolist()}
    (output / "completed.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if fit_steps:
        # 固定小组真实观测验证训练是否能拟合已知动作；沿用GBS256/micro8，不修改正式模型。
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config.sft_lr, weight_decay=0)
        rng = np.random.default_rng(314)
        for step in range(1, fit_steps + 1):
            ids = rng.integers(len(raw), size=config.global_batch_size)
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for start in range(0, len(ids), config.micro_batch_size):
                selected = ids[start : start + config.micro_batch_size]
                loss = policy.supervised_loss(
                    [raw[i] for i in selected],
                    torch.tensor([metadata[i]["length"] for i in selected]),
                )
                (loss / config.gas).backward()
                total += float(loss.detach()) / config.gas
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            row = {
                "step": step,
                "loss": total,
                "grad_norm": float(norm),
                "global_batch_size": config.global_batch_size,
                "micro_batch_size": config.micro_batch_size,
            }
            with (output / "sft-fit.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if step % 5 == 0:
                print(json.dumps(row), flush=True)
        # 使用与拟合前一致的噪声重新采样，排除训练loss下降但采样动作不变的实现错误。
        predictions = []
        for start in range(0, len(raw), config.micro_batch_size):
            torch.manual_seed(12000 + start)
            predictions.extend(
                policy.sample(raw[start : start + config.micro_batch_size], stochastic=False)[
                    "actions"
                ][:, 0].tolist()
            )
        error = np.array(predictions) - np.array([r["target"] for r in rows])
        result = {
            "steps": fit_steps,
            "mae_before": summary["normal"]["mae"],
            "mae_after": np.abs(error).mean(0).tolist(),
            "predictions": predictions,
        }
        assert all(
            p.grad is None for p in policy.model.paligemma_with_expert.paligemma.parameters()
        )
        (output / "fit-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "predictions"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-steps", type=int, default=0)
    args = parser.parse_args()
    check_critical_frames(args.config, args.checkpoint, args.output, args.fit_steps)
