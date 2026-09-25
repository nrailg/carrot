from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from openpi.models import model as openpi_model
from openpi.models import pi0_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jax-golden", type=Path, required=True)
    parser.add_argument("--openpi-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weight_path = args.checkpoint / "model.safetensors"
    assert weight_path.is_file(), f"OpenPI PyTorch checkpoint does not exist: {weight_path}"

    device = torch.device("cuda")
    with np.load(args.jax_golden, allow_pickle=False) as archive:
        jax_metadata = json.loads(archive["metadata_json"].item())
        images = [
            torch.from_numpy(image).permute(0, 3, 1, 2).contiguous().to(device)
            for image in archive["images"]
        ]
        image_masks = [torch.from_numpy(mask).to(device) for mask in archive["image_masks"]]
        tokens = torch.from_numpy(archive["tokens"]).to(device=device, dtype=torch.long)
        token_masks = torch.from_numpy(archive["token_masks"]).to(device)
        state = torch.from_numpy(archive["state"]).to(device)
        noise = torch.from_numpy(archive["noise"]).to(device)

    config = pi0_config.Pi0Config(pi05=True, pytorch_compile_mode=None)
    model = PI0Pytorch(config)
    safetensors.torch.load_model(model, weight_path)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model = model.to(device).eval()

    observation = openpi_model.Observation(
        images=dict(zip(IMAGE_KEYS, images, strict=True)),
        image_masks=dict(zip(IMAGE_KEYS, image_masks, strict=True)),
        state=state,
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_masks,
    )
    processed = model._preprocess_observation(observation, train=False)
    processed_images, processed_masks, processed_tokens, processed_token_masks, processed_state = (
        processed
    )

    with torch.no_grad():
        prefix_embeddings, _, _ = model.embed_prefix(
            processed_images, processed_masks, processed_tokens, processed_token_masks
        )
        suffix_embeddings, _, _, adarms_cond = model.embed_suffix(
            processed_state,
            noise,
            torch.ones((noise.shape[0],), device=device),
        )
        one_step = model.sample_actions(device, observation, noise=noise.clone(), num_steps=1)
        first_v_t = noise - one_step

    pi05 = model.paligemma_with_expert
    paligemma = pi05.paligemma.model
    vision_embeddings = paligemma.vision_tower.vision_model.embeddings
    weights = {
        "language_embedding_rows": paligemma.language_model.embed_tokens.weight[1:65],
        "action_in_kernel": model.action_in_proj.weight.T,
        "action_in_bias": model.action_in_proj.bias,
        "image_patch_kernel": vision_embeddings.patch_embedding.weight.permute(2, 3, 1, 0),
        "image_patch_bias": vision_embeddings.patch_embedding.bias,
        "image_position_embedding": vision_embeddings.position_embedding.weight[None],
        "image_head_kernel": paligemma.multi_modal_projector.linear.weight.T,
        "image_head_bias": paligemma.multi_modal_projector.linear.bias,
    }
    metadata = {
        "schema_version": 1,
        "openpi_commit": args.openpi_commit,
        "jax_openpi_commit": jax_metadata["openpi_commit"],
        "checkpoint": str(args.checkpoint.resolve()),
        "config": "Pi0Config(pi05=True, pytorch_compile_mode=None)",
        "steps": [1],
        "weight_dtypes": {name: str(weight.dtype) for name, weight in weights.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **{f"openpi_pytorch_{name}": _to_numpy(weight) for name, weight in weights.items()},
        openpi_pytorch_prefix_embeddings=_to_numpy(prefix_embeddings),
        openpi_pytorch_suffix_embeddings=_to_numpy(suffix_embeddings),
        openpi_pytorch_adarms_cond=_to_numpy(adarms_cond),
        openpi_pytorch_first_v_t=_to_numpy(first_v_t),
        openpi_pytorch_one_step=_to_numpy(one_step),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"golden={args.output}")


if __name__ == "__main__":
    main()
