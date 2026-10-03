"""Trace full-forward and cached inference on the newly trained fixed-noise case."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, override

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from carrot.data.so101 import build_dataset
from carrot.models.pi05 import transforms
from carrot.models.pi05.embodiments.so101 import create_so101_transform_spec
from carrot.models.pi05.model import PI0Pytorch
from carrot.models.pi05.model.pi0_pytorch import make_att_2d_masks
from recipes.pi05_sft_so101_fit_validation.fit import DropVision, fixed_noise


class AuditModel(PI0Pytorch):
    @override
    def _preprocess_observation(self, observation: Any, *, train: bool = True) -> Any:
        return super()._preprocess_observation(observation, train=False)


@torch.no_grad()
def _trace(model: AuditModel, transformed: dict, noise: np.ndarray,
           spec: transforms.Pi05TransformSpec, *, bf16_inputs: bool) -> dict:
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if bf16_inputs else torch.float32
    observation = transforms.to_observation(transformed, device=device, dtype=dtype)
    actions = torch.as_tensor(transformed["actions"], device=device, dtype=dtype)[None]
    latent = torch.as_tensor(noise, device=device, dtype=dtype)[None]
    images, masks, tokens, token_masks, state = model._preprocess_observation(observation)
    prefix, pad_masks, att_masks = model.embed_prefix(images, masks, tokens, token_masks)
    attention = model._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
    positions = torch.cumsum(pad_masks, dim=1) - 1
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=attention, position_ids=positions, past_key_values=None,
        inputs_embeds=[prefix, None], use_cache=True,
    )
    velocities = []
    handle = model.action_out_proj.register_forward_hook(
        lambda module, inputs, output: velocities.append(output.detach().clone())
    )
    rows = []
    try:
        for t in [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.01]:
            time = torch.tensor([t], device=device, dtype=dtype)
            errors = model(observation, actions, noise=latent, time=time)
            full = velocities[-1]
            x_t = time[:, None, None] * latent + (1 - time[:, None, None]) * actions
            cached = model.denoise_step(state, pad_masks, cache, x_t, time)
            np.testing.assert_allclose(errors.float().cpu().numpy(),
                (latent - actions - full).square().float().cpu().numpy(), rtol=0, atol=0)
            rows.append({"time_requested": t, "time_actual": float(time.item()),
                         "active_axis_mse": errors.float().mean(dim=(0, 1))[:6].cpu().tolist(),
                         "padded_axis_mean_mse": float(errors.float()[..., 6:].mean()),
                         "full_cached_max_abs": float((full.float() - cached.float()).abs().max())})
        rollouts = []
        for nfe in [10, 50]:
            x_t = latent.float()
            for step in range(nfe):
                time = torch.tensor([1 - step / nfe], device=device, dtype=dtype)
                velocity = model.denoise_step(state, pad_masks, cache, x_t.to(dtype), time)
                x_t = x_t - velocity.float() / nfe
            decoded = transforms.compose(spec.outputs)({"actions": x_t, "state": state})["actions"]
            rollouts.append({"nfe": nfe, "actions": decoded[0].float().cpu().tolist()})
        x_t = latent.float()
        for step in range(10):
            time = torch.tensor([1 - step / 10], device=device, dtype=dtype)
            suffix, suffix_pad, suffix_att, cond = model.embed_suffix(state, x_t.to(dtype), time)
            joint_pad = torch.cat([pad_masks, suffix_pad], dim=1)
            joint_att = torch.cat([att_masks, suffix_att], dim=1)
            joint_attention = model._prepare_attention_masks_4d(
                make_att_2d_masks(joint_pad, joint_att)
            )
            (_, output), _ = model.paligemma_with_expert.forward(
                attention_mask=joint_attention, position_ids=torch.cumsum(joint_pad, dim=1) - 1,
                past_key_values=None, inputs_embeds=[prefix.to(dtype), suffix.to(dtype)],
                use_cache=False, adarms_cond=[None, cond],
            )
            velocity = model.action_out_proj(output[:, -10:].to(dtype))
            x_t = x_t - velocity.float() / 10
        decoded = transforms.compose(spec.outputs)({"actions": x_t, "state": state})["actions"]
        rollouts.append({"nfe": 10, "path": "full_forward_without_cache",
                         "actions": decoded[0].float().cpu().tolist()})
    finally:
        handle.remove()
    return {"weight_dtype": str(model.action_in_proj.weight.dtype),
            "input_dtype": str(dtype), "teacher_points": rows, "rollouts": rollouts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((args.case_dir / "train.yaml").read_text())
    assert cfg["fit"] == {"vision": False, "noise_seed": 1053}
    kwargs = dict(cfg["dataset"]["factory_kwargs"])
    frame = kwargs.pop("frame_index")
    kwargs.pop("repeat_count")
    source = build_dataset(**kwargs)
    sample = source.dataset[frame]
    checkpoint = Path(cfg["output_dir"]) / "checkpoints/step-00001000"
    stats = {"state": source.state_stats, "actions": source.action_stats}
    exported = json.loads((checkpoint / "norm_stats.json").read_text())
    for key, exported_key in [("state", "state"), ("actions", "action")]:
        for name, value in stats[key].items():
            np.testing.assert_array_equal(value, exported[exported_key][name])
    model = AuditModel.from_pretrained(checkpoint).to("cuda:0").eval()
    assert model.config.action_horizon == kwargs["action_horizon"] == 10
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["tokenizer_path"], local_files_only=True, fix_mistral_regex=True
    )
    spec = create_so101_transform_spec(tokenizer, stats, model_action_dim=32,
                                       discrete_state_input=model.config.discrete_state_input)
    spec = replace(spec, inputs=spec.inputs + (DropVision(),))
    transformed = transforms.compose(spec.inputs)(sample)
    request = {k: v for k, v in sample.items() if k not in ("actions", "action_is_pad")}
    inference_input = transforms.compose(spec.inputs)(request)
    for key in ["state", "tokenized_prompt", "tokenized_prompt_mask"]:
        np.testing.assert_array_equal(transformed[key], inference_input[key])
    assert not any(transformed["image_mask"].values())
    restored = transforms.compose(spec.outputs)({"actions": transformed["actions"]})["actions"]
    np.testing.assert_allclose(restored, sample["actions"], atol=2e-5, rtol=0)
    noise = fixed_noise(1053, 10, 32)
    results = {"frame": frame, "checkpoint": str(checkpoint), "noise_seed": 1053,
               "checkpoint_config": json.loads((checkpoint / "config.json").read_text()),
               "train_eval_state_and_tokens": "exact match", "stats": "exact match",
               "action_transform_roundtrip_max_abs": float(
                   np.abs(restored - sample["actions"]).max()
               )}
    results["native_loaded_fp32"] = _trace(model, transformed, noise, spec, bf16_inputs=False)
    original_buffers = {name: value.clone() for name, value in model.named_buffers()}
    model.to(dtype=torch.bfloat16)
    results["bf16_weights_and_inputs_diagnostic"] = _trace(
        model, transformed, noise, spec, bf16_inputs=True
    )
    for name, value in original_buffers.items():
        parent, _, attribute = name.rpartition(".")
        model.get_submodule(parent).register_buffer(attribute, value, persistent=False)
    results["bf16_parameters_original_buffers_diagnostic"] = _trace(
        model, transformed, noise, spec, bf16_inputs=True
    )
    results["precision_scope"] = (
        "Single-device diagnostics; BF16 parameters/original buffers approximates FSDP compute "
        "but does not reproduce distributed wrapping or training augmentation."
    )
    for label in ["native_loaded_fp32", "bf16_weights_and_inputs_diagnostic",
                  "bf16_parameters_original_buffers_diagnostic"]:
        for row in results[label]["rollouts"]:
            error = np.asarray(row["actions"], dtype=np.float32) - sample["actions"]
            row["joint_mae"] = np.abs(error).mean(axis=0).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists()
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False))
    print("FIXED_FRAME_INFERENCE_AUDIT", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
