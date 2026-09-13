from pathlib import Path
from typing import Any

import pytest

from carrot.trainer.sft.config import SFTConfig


def _config(tmp_path: Path, values: dict[str, Any] | None = None) -> SFTConfig:
    del tmp_path
    values = dict(values or {})
    model = dict(values.get("model") or {})
    model.setdefault("tokenizer_path", "tokenizer")
    values["model"] = model
    return SFTConfig.from_dict(values)


def test_sft_config_builds_nested_configs(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        {
            "model": {"path": "model"},
            "dataset": {"repo_id": "dataset", "num_workers": 0},
            "optimizer": {"learning_rate": 2e-4, "betas": [0.8, 0.9]},
            "fsdp": {"param_dtype": "float32"},
            "steps": 2,
        },
    )

    assert config.model.path == "model"
    assert config.model.tokenizer_path == "tokenizer"
    assert config.dataset.num_workers == 0
    assert len(config.dataset.image_keys) == 3
    assert config.optimizer.betas == (0.8, 0.9)
    assert config.optimizer.weight_decay == 1e-10
    assert config.optimizer.max_grad_norm == 1.0
    assert config.fsdp.param_dtype == "float32"
    assert config.num_nodes == 1
    assert config.wandb.enabled is False


def test_sft_config_keeps_wandb_entity_as_string(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        {
            "wandb": {"enabled": True, "entity": 1001},
            "global_batch_size": 16,
            "dp_size": 16,
            "num_nodes": 2,
        },
    )

    assert config.wandb.entity == "1001"
    assert config.gpus_per_node == 8
    assert config.gas == 1


def test_sft_config_derives_gas(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        {"micro_batch_size": 2, "global_batch_size": 32, "dp_size": 8},
    )

    assert config.gas == 2


def test_sft_config_rejects_indivisible_global_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="global_batch_size"):
        _config(tmp_path, {"micro_batch_size": 3, "global_batch_size": 8, "dp_size": 2})


def test_sft_config_rejects_uneven_gpu_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divisible"):
        _config(tmp_path, {"dp_size": 16, "num_nodes": 3})


def test_sft_config_rejects_unknown_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown DatasetConfig fields"):
        _config(tmp_path, {"dataset": {"silent_typo": True}})


def test_sft_config_accepts_pi05_data_options(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        {
            "dataset": {
                "norm_stats_path": "/stats.json",
                "adapt_aloha": False,
                "delta_actions": False,
            }
        },
    )

    assert config.dataset.norm_stats_path == "/stats.json"
    assert config.dataset.adapt_aloha is False
    assert config.dataset.delta_actions is False


def test_sft_config_rejects_unsupported_dtype(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported param_dtype"):
        _config(tmp_path, {"fsdp": {"param_dtype": "float16"}})


def test_sft_config_uses_pi05_defaults() -> None:
    assert SFTConfig.from_dict({}).model.path == "Miical/pi05-base"


def test_sft_config_rejects_empty_tokenizer_path() -> None:
    with pytest.raises(ValueError, match="tokenizer_path"):
        SFTConfig.from_dict({"model": {"tokenizer_path": ""}})


def test_sft_config_rejects_decay_lr_above_peak(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="decay_learning_rate"):
        _config(
            tmp_path,
            {"optimizer": {"learning_rate": 1e-4, "decay_learning_rate": 2e-4}},
        )
