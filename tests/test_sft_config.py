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
    assert config.optimizer.betas == (0.8, 0.9)
    assert config.optimizer.weight_decay == 1e-10
    assert config.optimizer.max_grad_norm == 10.0
    assert config.fsdp.param_dtype == "float32"


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
