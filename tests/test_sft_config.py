import pytest

from carrot.trainer.sft.config import SFTConfig


def test_sft_config_builds_nested_configs() -> None:
    config = SFTConfig.from_dict(
        {
            "model": {"path": "model"},
            "dataset": {"repo_id": "dataset", "num_workers": 0},
            "optimizer": {"learning_rate": 2e-4, "betas": [0.8, 0.9]},
            "fsdp": {"param_dtype": "float32"},
            "steps": 2,
        }
    )

    assert config.model.path == "model"
    assert config.dataset.num_workers == 0
    assert config.dataset.rename_map == {}
    assert config.optimizer.betas == (0.8, 0.9)
    assert config.optimizer.weight_decay == 1e-10
    assert config.optimizer.max_grad_norm == 10.0
    assert config.fsdp.param_dtype == "float32"
    assert config.num_nodes == 1
    assert config.wandb.enabled is False


def test_sft_config_keeps_wandb_entity_as_string() -> None:
    config = SFTConfig.from_dict(
        {"wandb": {"enabled": True, "entity": 1001}, "num_gpus": 16, "num_nodes": 2}
    )

    assert config.wandb.entity == "1001"
    assert config.gpus_per_node == 8


def test_sft_config_rejects_uneven_gpu_split() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SFTConfig.from_dict({"num_gpus": 16, "num_nodes": 3})


def test_sft_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown DatasetConfig fields"):
        SFTConfig.from_dict({"dataset": {"silent_typo": True}})


def test_sft_config_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match="unsupported param_dtype"):
        SFTConfig.from_dict({"fsdp": {"param_dtype": "float16"}})


def test_sft_config_rejects_decay_lr_above_peak() -> None:
    with pytest.raises(ValueError, match="decay_learning_rate"):
        SFTConfig.from_dict(
            {"optimizer": {"learning_rate": 1e-4, "decay_learning_rate": 2e-4}}
        )
