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
    # 自定义数据参数应保留，未覆盖的工厂应指向 RoboTwin 模块。
    config = _config(
        tmp_path,
        {
            "model": {"path": "model"},
            "dataset": {"factory_kwargs": {"repo_id": "dataset"}, "num_workers": 0},
            "optimizer": {"learning_rate": 2e-4, "betas": [0.8, 0.9]},
            "fsdp": {"param_dtype": "float32"},
            "steps": 2,
        },
    )

    # 检查覆盖值与其余默认值同时保留。
    assert config.model.path == "model"
    assert config.model.tokenizer_path == "tokenizer"
    assert config.dataset.num_workers == 0
    assert config.dataset.factory == "carrot.data.robotwin.build_dataset"
    assert config.dataset.factory_kwargs["repo_id"] == "dataset"
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
    # 全局批次无法按微批次和数据并行数整除时，配置必须立即断言失败。
    with pytest.raises(AssertionError, match="global_batch_size"):
        _config(tmp_path, {"micro_batch_size": 3, "global_batch_size": 8, "dp_size": 2})


def test_sft_config_rejects_uneven_gpu_split(tmp_path: Path) -> None:
    # 数据并行数无法均分到节点时，配置必须立即断言失败。
    with pytest.raises(AssertionError, match="divisible"):
        _config(tmp_path, {"dp_size": 16, "num_nodes": 3})


def test_sft_config_rejects_unknown_fields(tmp_path: Path) -> None:
    # 未知字段通常是配置拼写错误，必须由嵌套配置解析器断言拒绝。
    with pytest.raises(AssertionError, match="unknown DatasetConfig fields"):
        _config(tmp_path, {"dataset": {"silent_typo": True}})


def test_sft_config_accepts_custom_dataset_integrations(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        {
            "dataset": {
                "norm_stats_path": "/stats.json",
                "factory": "example.datasets.build",
                "factory_kwargs": {"split": "train"},
                "preprocess": "example.transforms.preprocess",
            }
        },
    )

    assert config.dataset.norm_stats_path == "/stats.json"
    assert config.dataset.factory == "example.datasets.build"
    assert config.dataset.factory_kwargs == {"split": "train"}
    assert config.dataset.preprocess == "example.transforms.preprocess"


@pytest.mark.parametrize("asset_id", ["/absolute", "../outside"])
def test_sft_config_rejects_escaping_stats_asset_id(tmp_path: Path, asset_id: str) -> None:
    # asset id 决定 checkpoint 写入目录，必须阻止绝对路径和向上跳出。
    # 从普通配置入口构造，确认错误在保存前就被拒绝。
    with pytest.raises(AssertionError, match="norm_stats_asset_id"):
        _config(tmp_path, {"dataset": {"norm_stats_asset_id": asset_id}})


def test_sft_config_rejects_unsupported_dtype(tmp_path: Path) -> None:
    # FSDP 只支持训练路径定义的两种参数 dtype，其他取值必须断言失败。
    with pytest.raises(AssertionError, match="unsupported param_dtype"):
        _config(tmp_path, {"fsdp": {"param_dtype": "float16"}})


def test_sft_config_uses_pi05_defaults() -> None:
    # 默认配置的工厂与转换函数应在同一个 RoboTwin 模块中。
    config = SFTConfig.from_dict({})

    # 核对实际写入配置对象的默认路径。
    assert config.model.path == "Miical/pi05-base"
    assert config.dataset.factory == "carrot.data.robotwin.build_dataset"
    assert config.dataset.preprocess == "carrot.data.robotwin.robotwin_preprocess"


def test_sft_config_rejects_empty_tokenizer_path() -> None:
    # tokenizer 是模型构造的必需输入，空路径必须在配置阶段断言失败。
    with pytest.raises(AssertionError, match="tokenizer_path"):
        SFTConfig.from_dict({"model": {"tokenizer_path": ""}})


def test_sft_config_rejects_decay_lr_above_peak(tmp_path: Path) -> None:
    # 衰减终值不能高于峰值，违反学习率区间时必须断言失败。
    with pytest.raises(AssertionError, match="decay_learning_rate"):
        _config(
            tmp_path,
            {"optimizer": {"learning_rate": 1e-4, "decay_learning_rate": 2e-4}},
        )
