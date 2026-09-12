from pathlib import Path

import pytest

from carrot.models.smolvla.modeling import (
    has_processor_assets,
    parse_lerobot_version,
    processor_kwargs,
    require_lerobot_version,
)


def test_parse_lerobot_version_accepts_pep440_suffixes() -> None:
    assert parse_lerobot_version("0.6.1") == (0, 6, 1)
    assert parse_lerobot_version("0.6.1+cu129") == (0, 6, 1)
    assert parse_lerobot_version("0.6.2.dev1") == (0, 6, 2)


def test_require_lerobot_version_rejects_old_pin() -> None:
    with pytest.raises(ImportError, match="lerobot>=0.6.1"):
        require_lerobot_version("0.4.4")


def test_has_processor_assets_detects_local_and_hub_paths(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    ready = tmp_path / "ready"
    ready.mkdir()
    (ready / "policy_preprocessor.json").write_text("{}")

    assert has_processor_assets(str(ready))
    assert not has_processor_assets(str(empty))
    assert has_processor_assets("lerobot/smolvla_base")


def test_processor_kwargs_override_device_and_dataset_stats(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "policy_preprocessor.json").write_text("{}")
    stats = {"action": {"mean": 0}}

    kwargs = processor_kwargs(
        model_path=str(checkpoint),
        dataset_stats=stats,
        device="cuda:1",
        input_features={"observation.state": "state"},
        output_features={"action": "action"},
        normalization_mapping={"STATE": "MEAN_STD"},
    )

    assert kwargs["pretrained_path"] == str(checkpoint)
    assert kwargs["preprocessor_overrides"]["device_processor"]["device"] == "cuda:1"
    assert kwargs["preprocessor_overrides"]["normalizer_processor"]["stats"] is stats
    assert kwargs["postprocessor_overrides"]["unnormalizer_processor"]["stats"] is stats


def test_processor_kwargs_include_rename_map() -> None:
    kwargs = processor_kwargs(
        model_path="lerobot/smolvla_base",
        dataset_stats={},
        device="cuda:0",
        input_features={},
        output_features={},
        normalization_mapping={},
        rename_map={"observation.images.cam_high": "observation.images.camera1"},
    )

    assert kwargs["preprocessor_overrides"]["rename_observations_processor"]["rename_map"] == {
        "observation.images.cam_high": "observation.images.camera1"
    }


def test_processor_kwargs_override_local_tokenizer() -> None:
    kwargs = processor_kwargs(
        model_path="lerobot/smolvla_base",
        dataset_stats={},
        device="cuda:0",
        input_features={},
        output_features={},
        normalization_mapping={},
        tokenizer_name="/ckpt/SmolVLM2-500M-Video-Instruct",
    )

    assert kwargs["preprocessor_overrides"]["tokenizer_processor"] == {
        "tokenizer_name": "/ckpt/SmolVLM2-500M-Video-Instruct"
    }


def test_processor_kwargs_skip_pretrained_path_without_assets(tmp_path: Path) -> None:
    checkpoint = tmp_path / "old_ckpt"
    checkpoint.mkdir()

    kwargs = processor_kwargs(
        model_path=str(checkpoint),
        dataset_stats={},
        device="cuda:0",
        input_features={},
        output_features={},
        normalization_mapping={},
    )

    assert "pretrained_path" not in kwargs
    assert kwargs["dataset_stats"] == {}
    assert kwargs["preprocessor_overrides"] == {"device_processor": {"device": "cuda:0"}}
