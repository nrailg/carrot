"""Load PI0.5 models and assemble embodiment-specific transform specs."""

import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from carrot.models.pi05.embodiments import (
    create_aloha_transform_spec,
    create_libero_transform_spec,
    create_so101_transform_spec,
)
from carrot.models.pi05.model import PI0Pytorch

from .policy import Pi05Policy


def create_robotwin_policy(
    checkpoint_dir: str | Path,
    *,
    device: str,
    tokenizer_path: str | Path | None = None,
    norm_stats_path: str | Path | None = None,
    num_steps: int = 10,
    default_prompt: str | None = None,
) -> Pi05Policy:
    """Load a Carrot PI0.5 checkpoint with the RoboTwin ALOHA contract.

    Parameters
    ----------
    checkpoint_dir : str | pathlib.Path
    device : str
    tokenizer_path : str | pathlib.Path | None
    norm_stats_path : str | pathlib.Path | None
        Defaults to ``checkpoint_dir/norm_stats.json``.
    num_steps : int
    default_prompt : str | None

    Returns
    -------
    Pi05Policy
    """
    checkpoint_dir = _validate_checkpoint(checkpoint_dir)
    stats_path = (
        Path(norm_stats_path)
        if norm_stats_path is not None
        else checkpoint_dir / "norm_stats.json"
    )
    tokenizer_dir = _resolve_tokenizer_dir(checkpoint_dir, tokenizer_path)
    norm_stats = _load_norm_stats(stats_path)
    tokenizer = _load_tokenizer(tokenizer_dir)
    model = PI0Pytorch.from_pretrained(checkpoint_dir)
    transform_spec = create_aloha_transform_spec(
        tokenizer,
        norm_stats,
        model_action_dim=model.config.action_dim,
        default_prompt=default_prompt,
    )
    return Pi05Policy(model, transform_spec, device=device, num_steps=num_steps)


def create_libero_policy(
    checkpoint_dir: str | Path,
    *,
    device: str,
    tokenizer_path: str | Path | None = None,
    norm_stats_path: str | Path | None = None,
    num_steps: int = 10,
    default_prompt: str | None = None,
) -> Pi05Policy:
    """Load a PI0.5 checkpoint with the official LIBERO Panda contract.

    Parameters
    ----------
    checkpoint_dir : str | pathlib.Path
    device : str
    tokenizer_path : str | pathlib.Path | None
    norm_stats_path : str | pathlib.Path | None
        Defaults to the official OpenPI asset location or checkpoint root.
    num_steps : int
    default_prompt : str | None

    Returns
    -------
    Pi05Policy
    """
    checkpoint_dir = _validate_checkpoint(checkpoint_dir)
    official_stats = (
        checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    )
    default_stats = (
        official_stats if official_stats.is_file() else checkpoint_dir / "norm_stats.json"
    )
    stats_path = (
        Path(norm_stats_path)
        if norm_stats_path is not None
        else default_stats
    )
    tokenizer_dir = _resolve_tokenizer_dir(checkpoint_dir, tokenizer_path)
    norm_stats = _load_norm_stats(stats_path)
    tokenizer = _load_tokenizer(tokenizer_dir)
    model = PI0Pytorch.from_pretrained(checkpoint_dir)
    assert model.config.action_horizon == 10 and model.config.action_dim >= 7, (
        "LIBERO requires action_horizon=10 and model action_dim >= 7"
    )
    transform_spec = create_libero_transform_spec(
        tokenizer,
        norm_stats,
        model_action_dim=model.config.action_dim,
        default_prompt=default_prompt,
    )
    return Pi05Policy(model, transform_spec, device=device, num_steps=num_steps)


def create_so101_policy(
    checkpoint_dir: str | Path,
    *,
    device: str,
    tokenizer_path: str | Path | None = None,
    norm_stats_path: str | Path | None = None,
    num_steps: int = 10,
    default_prompt: str | None = None,
) -> Pi05Policy:
    """Load a Carrot PI0.5 checkpoint with the SO101 joint-control contract.

    Parameters
    ----------
    checkpoint_dir : str | pathlib.Path
    device : str
    tokenizer_path : str | pathlib.Path | None
    norm_stats_path : str | pathlib.Path | None
        Defaults to ``checkpoint_dir/norm_stats.json``.
    num_steps : int
    default_prompt : str | None

    Returns
    -------
    Pi05Policy
    """
    checkpoint_dir = _validate_checkpoint(checkpoint_dir)
    stats_path = (
        Path(norm_stats_path) if norm_stats_path is not None else checkpoint_dir / "norm_stats.json"
    )
    tokenizer_dir = _resolve_tokenizer_dir(checkpoint_dir, tokenizer_path)
    norm_stats = _load_norm_stats(stats_path)
    tokenizer = _load_tokenizer(tokenizer_dir)
    model = PI0Pytorch.from_pretrained(checkpoint_dir)
    transform_spec = create_so101_transform_spec(
        tokenizer,
        norm_stats,
        model_action_dim=model.config.action_dim,
        discrete_state_input=model.config.discrete_state_input,
        default_prompt=default_prompt,
    )
    return Pi05Policy(model, transform_spec, device=device, num_steps=num_steps)


def _validate_checkpoint(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    for name in ("model.safetensors", "config.json"):
        assert (checkpoint_dir / name).is_file(), (
            f"missing checkpoint file: {checkpoint_dir / name}"
        )
    return checkpoint_dir


def _resolve_tokenizer_dir(
    checkpoint_dir: Path, tokenizer_path: str | Path | None
) -> Path:
    tokenizer_dir = checkpoint_dir
    if not (tokenizer_dir / "tokenizer_config.json").is_file():
        assert tokenizer_path is not None, (
            f"missing tokenizer_config.json: {checkpoint_dir / 'tokenizer_config.json'}"
        )
        tokenizer_dir = Path(tokenizer_path)
        assert (tokenizer_dir / "tokenizer_config.json").is_file(), (
            f"missing tokenizer_config.json: {tokenizer_dir / 'tokenizer_config.json'}"
        )
    return tokenizer_dir


def _load_tokenizer(tokenizer_dir: Path) -> Any:
    return AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        fix_mistral_regex=True,
    )


def _load_norm_stats(path: Path) -> dict[str, dict[str, Any]]:
    assert path.is_file(), f"missing normalization stats: {path}"
    with path.open() as stream:
        payload = json.load(stream)
    if "norm_stats" in payload:
        payload = payload["norm_stats"]
    action_key = "actions" if "actions" in payload else "action"
    return {"state": payload["state"], "actions": payload[action_key]}
