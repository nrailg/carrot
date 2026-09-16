"""Load a Carrot PI0.5 checkpoint and its RoboTwin inference assets."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from carrot.models.pi05.model import PI0Policy

from .policy import Pi05Policy


def create_trained_policy(
    checkpoint_dir: str | Path,
    *,
    device: str,
    tokenizer_path: str | Path | None = None,
    num_steps: int = 10,
    default_prompt: str | None = None,
) -> Pi05Policy:
    """Load an exported Carrot RoboTwin SFT checkpoint for local inference.

    Parameters
    ----------
    checkpoint_dir : str | pathlib.Path
        Directory containing model.safetensors, config.json and norm_stats.json.
        Use with checkpoints trained with robotwin_preprocess.
    device : str
    tokenizer_path : str | pathlib.Path | None
        Explicit local fallback for older exports without tokenizer assets.
        A tokenizer saved in the checkpoint takes precedence.
    num_steps : int
    default_prompt : str | None

    Returns
    -------
    Pi05Policy

    Raises
    ------
    FileNotFoundError
        A required checkpoint artifact is absent.
    """
    checkpoint_dir = Path(checkpoint_dir)
    for name in ("model.safetensors", "config.json", "norm_stats.json"):
        if not (checkpoint_dir / name).is_file():
            raise FileNotFoundError(checkpoint_dir / name)
    tokenizer_dir = checkpoint_dir
    if not (tokenizer_dir / "tokenizer_config.json").is_file():
        if tokenizer_path is None:
            raise FileNotFoundError(checkpoint_dir / "tokenizer_config.json")
        tokenizer_dir = Path(tokenizer_path)
        if not (tokenizer_dir / "tokenizer_config.json").is_file():
            raise FileNotFoundError(tokenizer_dir / "tokenizer_config.json")
    with (checkpoint_dir / "norm_stats.json").open() as stream:
        stats = json.load(stream)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, fix_mistral_regex=True
    )
    model = PI0Policy.from_pretrained(checkpoint_dir)
    return Pi05Policy(
        model,
        tokenizer,
        {"state": stats["state"], "actions": stats["action"]},
        device=device,
        num_steps=num_steps,
        default_prompt=default_prompt,
    )
