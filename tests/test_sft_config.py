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
    assert config.optimizer.betas == [0.8, 0.9]
    assert config.fsdp.param_dtype == "float32"


def test_sft_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown DatasetConfig fields"):
        SFTConfig.from_dict({"dataset": {"silent_typo": True}})
