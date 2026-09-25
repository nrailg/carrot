from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as openpi_model
from openpi.models import pi0_config
from openpi.shared import nnx_utils

SEED = 20260914
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(SEED)
    images = generator.random((3, 1, 224, 224, 3), dtype=np.float32) * 2 - 1
    image_masks = np.ones((3, 1), dtype=np.bool_)
    tokens = np.arange(1, 65, dtype=np.int32)[None, :]
    token_masks = np.ones_like(tokens, dtype=np.bool_)
    state = np.zeros((1, 32), dtype=np.float32)
    noise = generator.standard_normal((1, 50, 32)).astype(np.float32)
    return images, image_masks, tokens, token_masks, state, noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--openpi-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    params_path = args.checkpoint / "params"
    assert params_path.is_dir(), f"OpenPI params directory does not exist: {params_path}"

    images, image_masks, tokens, token_masks, state, noise = _inputs()
    observation = openpi_model.Observation(
        images={key: jnp.asarray(images[index]) for index, key in enumerate(IMAGE_KEYS)},
        image_masks={key: jnp.asarray(image_masks[index]) for index, key in enumerate(IMAGE_KEYS)},
        state=jnp.asarray(state),
        tokenized_prompt=jnp.asarray(tokens),
        tokenized_prompt_mask=jnp.asarray(token_masks),
    )

    config = pi0_config.Pi0Config(pi05=True)
    params = openpi_model.restore_params(params_path, dtype=jnp.bfloat16)
    model = config.load(params)
    processed_observation = openpi_model.preprocess_observation(None, observation, train=False)
    prefix_embeddings, _, _ = model.embed_prefix(processed_observation)
    suffix_embeddings, _, _, adarms_cond = model.embed_suffix(
        processed_observation,
        jnp.asarray(noise),
        jnp.ones((noise.shape[0],), dtype=jnp.float32),
    )
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    rng = jax.random.key(SEED)
    one_step = sample_actions(rng, observation, noise=jnp.asarray(noise), num_steps=1)
    first_v_t = jnp.asarray(noise) - one_step

    metadata = {
        "schema_version": 3,
        "seed": SEED,
        "openpi_commit": args.openpi_commit,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": "Pi0Config(pi05=True)",
        "parameter_dtype": "bfloat16",
        "image_layout": "NIHWC",
        "steps": [1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        images=images,
        image_masks=image_masks,
        tokens=tokens,
        token_masks=token_masks,
        state=state,
        noise=noise,
        openpi_language_embedding_rows=np.asarray(
            params["PaliGemma"]["llm"]["embedder"]["input_embedding"][1:65],
            dtype=np.float32,
        ),
        openpi_action_in_kernel=np.asarray(params["action_in_proj"]["kernel"], dtype=np.float32),
        openpi_action_in_bias=np.asarray(params["action_in_proj"]["bias"], dtype=np.float32),
        openpi_image_patch_kernel=np.asarray(
            params["PaliGemma"]["img"]["embedding"]["kernel"], dtype=np.float32
        ),
        openpi_image_patch_bias=np.asarray(
            params["PaliGemma"]["img"]["embedding"]["bias"], dtype=np.float32
        ),
        openpi_image_position_embedding=np.asarray(
            params["PaliGemma"]["img"]["pos_embedding"], dtype=np.float32
        ),
        openpi_image_head_kernel=np.asarray(
            params["PaliGemma"]["img"]["head"]["kernel"], dtype=np.float32
        ),
        openpi_image_head_bias=np.asarray(
            params["PaliGemma"]["img"]["head"]["bias"], dtype=np.float32
        ),
        openpi_prefix_embeddings=np.asarray(jax.device_get(prefix_embeddings), dtype=np.float32),
        openpi_suffix_embeddings=np.asarray(jax.device_get(suffix_embeddings), dtype=np.float32),
        openpi_adarms_cond=np.asarray(jax.device_get(adarms_cond), dtype=np.float32),
        openpi_first_v_t=np.asarray(jax.device_get(first_v_t), dtype=np.float32),
        openpi_one_step=np.asarray(jax.device_get(one_step), dtype=np.float32),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"golden={args.output}")


if __name__ == "__main__":
    main()
