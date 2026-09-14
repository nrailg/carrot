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
    if not params_path.is_dir():
        raise FileNotFoundError(f"OpenPI params directory does not exist: {params_path}")

    images, image_masks, tokens, token_masks, state, noise = _inputs()
    observation = openpi_model.Observation(
        images={key: jnp.asarray(images[index]) for index, key in enumerate(IMAGE_KEYS)},
        image_masks={
            key: jnp.asarray(image_masks[index]) for index, key in enumerate(IMAGE_KEYS)
        },
        state=jnp.asarray(state),
        tokenized_prompt=jnp.asarray(tokens),
        tokenized_prompt_mask=jnp.asarray(token_masks),
    )

    config = pi0_config.Pi0Config(pi05=True)
    params = openpi_model.restore_params(params_path, dtype=jnp.bfloat16)
    model = config.load(params)
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    rng = jax.random.key(SEED)
    one_step = sample_actions(rng, observation, noise=jnp.asarray(noise), num_steps=1)
    ten_steps = sample_actions(rng, observation, noise=jnp.asarray(noise), num_steps=10)

    metadata = {
        "schema_version": 1,
        "seed": SEED,
        "openpi_commit": args.openpi_commit,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": "Pi0Config(pi05=True)",
        "parameter_dtype": "bfloat16",
        "image_layout": "NIHWC",
        "steps": [1, 10],
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
        openpi_one_step=np.asarray(jax.device_get(one_step), dtype=np.float32),
        openpi_ten_steps=np.asarray(jax.device_get(ten_steps), dtype=np.float32),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"golden={args.output}")


if __name__ == "__main__":
    main()
