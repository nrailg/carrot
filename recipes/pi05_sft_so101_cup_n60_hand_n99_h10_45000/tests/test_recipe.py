"""Offline checks for the pinned recipe, isolated run files, and shell CLI forwarding."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import yaml

from carrot.data.loading import load_callable
from recipes.pi05_sft_so101_cup_n60_hand_n99_h10_45000 import configure as configure_module
from recipes.pi05_sft_so101_cup_n60_hand_n99_h10_45000 import preflight

HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = HERE.parents[1]
RECIPE = "recipes.pi05_sft_so101_cup_n60_hand_n99_h10_45000"
NORM_SHA256 = "ed6087fd3e59c7e103a295abe2811241d7b30f5ae88975258d348e4a747dac0b"


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    data_root.mkdir()
    return {
        "data_root": data_root,
        "model_path": tmp_path / "model-on-training-host",
        "tokenizer_path": tmp_path / "tokenizer-on-training-host",
        "run_dir": tmp_path / "runs/new-run",
    }


def test_configure_retains_training_and_data_contract(paths: dict[str, Path]) -> None:
    config_path = configure_module.configure(**paths)
    config = yaml.safe_load(config_path.read_text())
    assert config_path == paths["run_dir"] / "train.yaml"
    assert config["model"] == {
        "path": str(paths["model_path"]),
        "tokenizer_path": str(paths["tokenizer_path"]),
    }
    assert {
        key: config[key]
        for key in (
            "steps",
            "save_freq",
            "log_freq",
            "seed",
            "micro_batch_size",
            "global_batch_size",
            "dp_size",
            "num_nodes",
        )
    } == {
        "steps": 45000,
        "save_freq": 5000,
        "log_freq": 1,
        "seed": 1000,
        "micro_batch_size": 4,
        "global_batch_size": 64,
        "dp_size": 8,
        "num_nodes": 1,
    }
    assert config["optimizer"] == {
        "learning_rate": 1e-6,
        "weight_decay": 1e-10,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "warmup_steps": 100,
        "decay_steps": 45000,
        "decay_learning_rate": 1e-6,
        "max_grad_norm": 1.0,
    }
    assert config["fsdp"] == {
        "enabled": True,
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
        "reshard_after_forward": True,
    }
    assert config["wandb"] == {"enabled": False}
    assert config["output_dir"] == str(paths["run_dir"] / "model")
    assert not (paths["run_dir"] / "model").exists()
    dataset = config["dataset"]
    assert dataset["factory"] == f"{RECIPE}.dataset.build_dataset"
    assert dataset["num_workers"] == 4
    assert dataset["preprocess"] is None
    assert dataset["norm_stats_source"] == "file"
    kwargs = dataset["factory_kwargs"]
    assert {
        key: kwargs[key]
        for key in (
            "action_horizon",
            "video_backend",
            "base_image_key",
            "wrist_image_key",
        )
    } == {
        "action_horizon": 10,
        "video_backend": "pyav",
        "base_image_key": "observation.images.top",
        "wrist_image_key": "observation.images.wrist",
    }
    catalog = json.loads((HERE / "inputs.json").read_text())
    assert (catalog["episodes"], catalog["frames"], catalog["action_horizon"]) == (159, 10179, 10)
    assert [source["repo_id"] for source in catalog["sources"]] == [
        "Jasonchen9/knock-over-hand-cream-n99-v1",
        "Jasonchen9/cup-to-white-square-multi-n60-v1",
    ]
    assert [source["revision"] for source in catalog["sources"]] == [
        "7a21f1318d37b6b602674682e451709b8661c541",
        "628274eaa5013d90216fdec4c6fd7c2b9939ea03",
    ]
    assert [(source["episodes"], source["frames"]) for source in catalog["sources"]] == [
        (99, 3447),
        (60, 6732),
    ]
    for configured, source in zip(kwargs["datasets"], catalog["sources"], strict=True):
        assert configured == {
            **{
                key: source[key]
                for key in (
                    "repo_id",
                    "revision",
                    "frames",
                    "episodes",
                    "manifest_sha256",
                )
            },
            "root": str(paths["data_root"] / source["directory"]),
        }
        assert f"https://huggingface.co/datasets/jason-so101/{source['directory']}" in (
            config_path.read_text()
        )
    norm_path = paths["run_dir"] / "assets/pi05_norm_stats.json"
    assert dataset["norm_stats_path"] == kwargs["norm_stats_path"] == str(norm_path)
    assert norm_path.read_bytes() == (HERE / "assets/pi05_norm_stats.json").read_bytes()
    assert hashlib.sha256(norm_path.read_bytes()).hexdigest() == NORM_SHA256
    assert norm_path.stat().st_ino != (HERE / "assets/pi05_norm_stats.json").stat().st_ino
    provenance = json.loads((paths["run_dir"] / "provenance.json").read_text())
    assert provenance["source_root"] == str(REPO_ROOT)
    assert provenance["norm_sha256"] == NORM_SHA256
    assert provenance["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()


def test_recipe_factory_is_importable() -> None:
    template = yaml.safe_load((HERE / "train.template.yaml").read_text())
    factory = load_callable(template["dataset"]["factory"])
    assert factory.__module__ == f"{RECIPE}.dataset"


def test_configuration_safely_quotes_unusual_paths(tmp_path: Path) -> None:
    root = tmp_path / "quoted: # ' \" \\ path\nnext line"
    root.mkdir()
    config_path = configure_module.configure(
        root, root / "model", root / "tokens", tmp_path / "run"
    )
    config = yaml.safe_load(config_path.read_text())
    assert config["model"]["path"] == str(root / "model")
    assert config["model"]["tokenizer_path"] == str(root / "tokens")
    assert config["dataset"]["factory_kwargs"]["datasets"][0]["root"] == str(
        root / "knock-over-hand-cream-n99-v1"
    )


@pytest.mark.parametrize("name", ["data_root", "model_path", "tokenizer_path", "run_dir"])
def test_relative_paths_are_rejected(paths: dict[str, Path], name: str) -> None:
    paths[name] = Path("relative")
    with pytest.raises(ValueError, match="absolute path"):
        configure_module.configure(**paths)


def test_existing_run_is_never_overwritten(paths: dict[str, Path]) -> None:
    config_path = configure_module.configure(**paths)
    contents = config_path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        configure_module.configure(**paths)
    assert config_path.read_bytes() == contents


@pytest.mark.parametrize("protected", ["source", "data_root", "model_path", "tokenizer_path"])
def test_run_inside_source_or_inputs_is_rejected(paths: dict[str, Path], protected: str) -> None:
    root = REPO_ROOT if protected == "source" else paths[protected]
    paths["run_dir"] = root / "must-not-be-created"
    with pytest.raises(ValueError, match="outside the source and input directories"):
        configure_module.configure(**paths)
    assert not paths["run_dir"].exists()


def test_run_path_symlinks_cannot_bypass_isolation(paths: dict[str, Path], tmp_path: Path) -> None:
    link = tmp_path / "data-link"
    link.symlink_to(paths["data_root"], target_is_directory=True)
    paths["run_dir"] = link / "new-run"
    with pytest.raises(ValueError, match="outside the source and input directories"):
        configure_module.configure(**paths)
    assert not paths["run_dir"].exists()


def test_changed_norm_is_rejected_before_creating_run(
    paths: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    altered_recipe = tmp_path / "altered-recipe"
    (altered_recipe / "assets").mkdir(parents=True)
    (altered_recipe / "assets/pi05_norm_stats.json").write_text("{}\n")
    monkeypatch.setattr(configure_module, "HERE", altered_recipe)
    with pytest.raises(ValueError, match="normalization SHA256"):
        configure_module.configure(**paths)
    assert not paths["run_dir"].exists()


def test_preflight_accepts_valid_hyperparameter_changes(paths: dict[str, Path]) -> None:
    config = yaml.safe_load(configure_module.configure(**paths).read_text())
    config.update(
        steps=30000,
        global_batch_size=32,
        micro_batch_size=2,
        dp_size=4,
        num_nodes=2,
        log_freq=10,
        save_freq=0,
        seed=42,
    )
    config["optimizer"].update(learning_rate=2e-6, warmup_steps=200, decay_steps=30000)
    config["fsdp"]["reshard_after_forward"] = False
    config["wandb"] = {"enabled": True, "project": "joint-recipe-test", "entity": "test"}
    config["dataset"]["num_workers"] = 0
    config["dataset"]["factory_kwargs"]["video_backend"] = "torchcodec"
    sources, _ = preflight.check_config(config)
    assert len(sources) == 2


@pytest.mark.parametrize("protected", ["source", "first_data", "second_data", "model", "tokenizer"])
@pytest.mark.parametrize("via_symlink", [False, True])
def test_edited_output_cannot_enter_protected_inputs(
    paths: dict[str, Path], tmp_path: Path, protected: str, via_symlink: bool
) -> None:
    config_path = configure_module.configure(**paths)
    config = yaml.safe_load(config_path.read_text())
    sources = config["dataset"]["factory_kwargs"]["datasets"]
    root = {
        "source": REPO_ROOT,
        "first_data": Path(sources[0]["root"]),
        "second_data": Path(sources[1]["root"]),
        "model": paths["model_path"],
        "tokenizer": paths["tokenizer_path"],
    }[protected]
    parent = root
    if via_symlink:
        parent = tmp_path / "input-alias"
        parent.symlink_to(root, target_is_directory=True)
    output = parent / f"must-not-be-created-{tmp_path.name}"
    config["output_dir"] = str(output)
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError):
        preflight.check_config(yaml.safe_load(config_path.read_text()))
    assert not output.exists()


def test_edited_output_can_use_a_new_external_directory(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    config = yaml.safe_load(configure_module.configure(**paths).read_text())
    output = tmp_path / "separate-experiment" / "model"
    config["output_dir"] = str(output)
    preflight.check_config(config)
    assert preflight.check_output_path(config) == output.resolve()
    assert not output.parent.exists()


def test_output_rejects_a_dangling_leaf_symlink(paths: dict[str, Path], tmp_path: Path) -> None:
    config = yaml.safe_load(configure_module.configure(**paths).read_text())
    output = Path(config["output_dir"])
    target = tmp_path / "uncreated-output-target"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        preflight.check_config(config)
    assert output.is_symlink()
    assert not target.exists()


def test_output_reservation_refuses_a_second_launch_without_removing_files(
    paths: dict[str, Path],
) -> None:
    config = yaml.safe_load(configure_module.configure(**paths).read_text())
    output = Path(config["output_dir"])
    assert preflight.reserve_output_directory(config) == output.resolve()
    marker = output / "first-launch.txt"
    marker.write_text("owned by the first launch")
    with pytest.raises((ValueError, FileExistsError)):
        preflight.reserve_output_directory(config)
    assert marker.read_text() == "owned by the first launch"


def test_concurrent_output_reservation_has_exactly_one_winner(
    paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = yaml.safe_load(configure_module.configure(**paths).read_text())
    output = Path(config["output_dir"])
    check_output = preflight.check_output_path
    both_checked = Barrier(2)

    def simultaneous_check(candidate):
        checked = check_output(candidate)
        # Force both launchers past the read-only existence check before either mkdir.
        both_checked.wait(timeout=10)
        return checked

    monkeypatch.setattr(preflight, "check_output_path", simultaneous_check)

    def launch(identifier):
        try:
            reserved = preflight.reserve_output_directory(config)
        except (ValueError, FileExistsError):
            return None
        (reserved / "owner.txt").write_text(str(identifier))
        return identifier

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(launch, identifier) for identifier in (1, 2)]
        results = [future.result(timeout=15) for future in futures]
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert (output / "owner.txt").read_text() == str(winners[0])


@pytest.mark.parametrize("dp_size", [1, 8])
def test_disabling_fsdp_is_only_supported_on_one_gpu(paths: dict[str, Path], dp_size: int) -> None:
    config = yaml.safe_load(configure_module.configure(**paths).read_text())
    config["dp_size"] = dp_size
    config["fsdp"]["enabled"] = False
    if dp_size == 1:
        preflight.check_config(config)
    else:
        with pytest.raises(ValueError, match="Multi-GPU training requires fsdp.enabled=true"):
            preflight.check_config(config)


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "steps", 0),
        (None, "global_batch_size", 63),
        (None, "num_nodes", 3),
        ("optimizer", "learning_rate", 0),
        ("dataset", "num_workers", -1),
        ("dataset", "norm_stats_source", "dataset"),
        ("dataset", "preprocess", "custom.preprocessing"),
        ("factory_kwargs", "action_horizon", 50),
        ("factory_kwargs", "base_image_key", "observation.images.wrist"),
    ],
)
def test_preflight_still_rejects_invalid_or_incompatible_configuration(
    paths: dict[str, Path], section: str | None, key: str, value: object
) -> None:
    config = yaml.safe_load(configure_module.configure(**paths).read_text())
    target = (
        config["dataset"]["factory_kwargs"]
        if section == "factory_kwargs"
        else config[section]
        if section is not None
        else config
    )
    target[key] = value
    with pytest.raises(ValueError):
        preflight.check_config(config)


@pytest.mark.parametrize("change", ["different_paths", "corrupt_bytes"])
def test_preflight_requires_one_unchanged_norm(
    paths: dict[str, Path],
    tmp_path: Path,
    change: str,
) -> None:
    config = yaml.safe_load(configure_module.configure(**paths).read_text())
    dataset = config["dataset"]
    if change == "different_paths":
        dataset["factory_kwargs"]["norm_stats_path"] = str(HERE / "assets/pi05_norm_stats.json")
        expected = "same joint normalization"
    else:
        corrupt_norm = tmp_path / "corrupt-norm.json"
        corrupt_norm.write_text("{}\n")
        dataset["norm_stats_path"] = str(corrupt_norm)
        dataset["factory_kwargs"]["norm_stats_path"] = str(corrupt_norm)
        expected = "Joint normalization differs"
    with pytest.raises(ValueError, match=expected):
        preflight.check_config(config)


def test_preflight_rejects_source_revision_changes(paths: dict[str, Path]) -> None:
    config = yaml.safe_load(configure_module.configure(**paths).read_text())
    config["dataset"]["factory_kwargs"]["datasets"][0]["revision"] = "different-release"
    with pytest.raises(ValueError, match="Source release parameter differs"):
        preflight.check_config(config)


@pytest.mark.parametrize("full", [False, True])
def test_both_validation_modes_check_release_contents(
    paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch, full: bool
) -> None:
    config_path = configure_module.configure(**paths)
    checked_roots, sample_modes = [], []

    def verify(root, source):
        checked_roots.append(root)
        return 12

    def samples(config, norm, *, full=False):
        sample_modes.append(full)
        return {
            "frames": 10179,
            "episodes": 159,
            "samples_checked": 477 if full else 6,
            "statistics_recomputed": full,
        }

    monkeypatch.setattr(preflight, "verify_release", verify)
    monkeypatch.setattr(preflight, "check_samples", samples)
    result = preflight.validate(config_path, check_model=False, full=full)
    assert checked_roots == [
        paths["data_root"] / "knock-over-hand-cream-n99-v1",
        paths["data_root"] / "cup-to-white-square-multi-n60-v1",
    ]
    assert sample_modes == [full]
    assert result["release_files_checked"] == 24
    assert result["full_validation"] is full
    assert result["statistics_recomputed"] is full
    assert result["model_config_checked"] is False
    assert result["ray_started"] is result["training_started"] is False
    assert result["output_reserved"] is False
    assert not (paths["run_dir"] / "model").exists()


@pytest.fixture
def reservable_run(paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch):
    """Use configuration-only model assets and mocked data checks, never model loading."""
    paths["model_path"].mkdir()
    paths["tokenizer_path"].mkdir()
    (paths["model_path"] / "config.json").write_text(
        json.dumps({"action_horizon": 10, "action_dim": 32})
    )
    config_path = configure_module.configure(**paths)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output_dir"])

    def verify(root, source):
        assert not output.exists()
        return 12

    def samples(candidate, norm, *, full=False):
        assert not output.exists()
        return {
            "frames": 10179,
            "episodes": 159,
            "samples_checked": 477 if full else 6,
            "statistics_recomputed": full,
        }

    monkeypatch.setattr(preflight, "verify_release", verify)
    monkeypatch.setattr(preflight, "check_samples", samples)
    return config_path, config


def test_default_validation_does_not_reserve_output(reservable_run) -> None:
    config_path, config = reservable_run
    report = preflight.validate(config_path)
    assert report["output_reserved"] is False
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("full", [False, True])
def test_validation_reserves_output_only_after_all_checks(reservable_run, full: bool) -> None:
    config_path, config = reservable_run
    report = preflight.validate(config_path, full=full, reserve_output=True)
    assert report["output_reserved"] is True
    assert report["training_started"] is report["ray_started"] is False
    assert Path(config["output_dir"]).is_dir()


@pytest.mark.parametrize("failure", ["model", "release", "samples"])
def test_failed_validation_does_not_reserve_output(
    reservable_run, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    config_path, config = reservable_run

    def fail_check(*args, **kwargs):
        raise ValueError("Simulated invalid release or samples")

    if failure == "model":
        (Path(config["model"]["path"]) / "config.json").write_text(
            json.dumps({"action_horizon": 50, "action_dim": 32})
        )
    else:
        target = "verify_release" if failure == "release" else "check_samples"
        monkeypatch.setattr(preflight, target, fail_check)
    with pytest.raises(ValueError):
        preflight.validate(config_path, reserve_output=True)
    assert not Path(config["output_dir"]).exists()


def test_data_only_validation_cannot_reserve_output(reservable_run) -> None:
    config_path, config = reservable_run
    with pytest.raises(ValueError):
        preflight.validate(config_path, check_model=False, reserve_output=True)
    assert not Path(config["output_dir"]).exists()


def test_configuration_changed_during_preflight_is_not_reserved(
    reservable_run, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, config = reservable_run
    check_samples = preflight.check_samples

    def edit_configuration(candidate, norm, *, full=False):
        report = check_samples(candidate, norm, full=full)
        config_path.write_text(config_path.read_text() + "\n# Edited during validation\n")
        return report

    monkeypatch.setattr(preflight, "check_samples", edit_configuration)
    with pytest.raises(ValueError, match="Configuration changed during preflight"):
        preflight.validate(config_path, reserve_output=True)
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("field,value", [("action_horizon", 50), ("action_dim", 6)])
def test_preflight_rejects_incompatible_model_configuration(
    paths: dict[str, Path], field: str, value: int
) -> None:
    paths["model_path"].mkdir()
    paths["tokenizer_path"].mkdir()
    model_config = {"action_horizon": 10, "action_dim": 32, field: value}
    (paths["model_path"] / "config.json").write_text(json.dumps(model_config))
    config_path = configure_module.configure(**paths)
    with pytest.raises(ValueError, match="H10 PI0.5"):
        preflight.validate(config_path)


def seal_tiny_release(root: Path, relative_paths: tuple[str, ...]) -> dict[str, str]:
    """Record expected bytes independently of the release verifier."""
    files = {
        relative: {
            "bytes": (root / relative).stat().st_size,
            "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
        }
        for relative in relative_paths
    }
    manifest = root / "release_manifest.json"
    manifest.write_text(json.dumps({"dataset_id": "test/tiny-release", "files": files}))
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((root / relative).read_bytes()).hexdigest()}  {relative}\n"
            for relative in (*files, "release_manifest.json")
        )
    )
    return {
        "repo_id": "test/tiny-release",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "norm_sha256": files["meta/pi05_norm_stats.json"]["sha256"],
    }


@pytest.fixture
def tiny_release(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "tiny-release"
    (root / "meta").mkdir(parents=True)
    (root / "meta/pi05_norm_stats.json").write_bytes(b"{}\n")
    (root / "payload.bin").write_bytes(b"abc")
    source = seal_tiny_release(root, ("meta/pi05_norm_stats.json", "payload.bin"))
    return root, source


def add_transport_files(root: Path, relative_paths: tuple[str, ...]) -> None:
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"transport metadata\n")


def test_preflight_detects_release_byte_corruption(
    tiny_release: tuple[Path, dict[str, str]],
) -> None:
    root, source = tiny_release
    assert preflight.verify_release(root, source) == 3
    (root / "payload.bin").write_bytes(b"abd")
    with pytest.raises(ValueError, match="Release content differs"):
        preflight.verify_release(root, source)


@pytest.mark.parametrize(
    "metadata",
    [
        (
            ".gitattributes",
            ".cache/huggingface/.gitignore",
            ".cache/huggingface/download/payload.bin.metadata",
            ".cache/huggingface/download/payload.bin.lock",
            ".cache/huggingface/download/subdir/chunk.incomplete",
        ),
        (".gitattributes", ".git/HEAD", ".git/config", ".git/objects/ab/cdef"),
        (".git",),
        (".DS_Store", "meta/.DS_Store"),
    ],
    ids=["hf-download", "git-clone", "git-worktree", "finder-metadata"],
)
def test_preflight_accepts_transport_metadata_without_modifying_it(
    tiny_release: tuple[Path, dict[str, str]], metadata: tuple[str, ...]
) -> None:
    root, source = tiny_release
    add_transport_files(root, metadata)
    before = {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    assert preflight.verify_release(root, source) == 3
    after = {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("change", ["payload", "missing", "norm", "manifest"])
def test_transport_metadata_does_not_hide_release_damage(
    tiny_release: tuple[Path, dict[str, str]], change: str
) -> None:
    root, source = tiny_release
    add_transport_files(
        root, (".gitattributes", ".cache/huggingface/download/payload.bin.metadata")
    )
    if change == "payload":
        (root / "payload.bin").write_bytes(b"abd")  # Same length, different content.
        error = "Release content differs"
    elif change == "missing":
        (root / "payload.bin").unlink()
        error = "Unexpected or missing release files"
    elif change == "norm":
        (root / "meta/pi05_norm_stats.json").write_bytes(b"[]\n")
        error = "Release content differs"
    else:
        manifest = root / "release_manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        error = "Release manifest differs"
    with pytest.raises(ValueError, match=error):
        preflight.verify_release(root, source)


@pytest.mark.parametrize(
    "unexpected",
    [
        "extra.bin",
        ".hidden",
        ".cache/other/download.bin",
        ".cache/huggingface-backup/state",
        "meta/.gitattributes",
        "meta/.cache/huggingface/state",
        "meta/.git/HEAD",
    ],
)
def test_preflight_rejects_files_outside_transport_allowlist(
    tiny_release: tuple[Path, dict[str, str]], unexpected: str
) -> None:
    root, source = tiny_release
    add_transport_files(root, (".cache/huggingface/download/payload.bin.metadata", unexpected))
    with pytest.raises(ValueError, match="Unexpected or missing release files"):
        preflight.verify_release(root, source)


@pytest.mark.parametrize(
    "relative", [".gitattributes", ".git/HEAD", ".cache/huggingface/tracked.bin"]
)
def test_manifest_entries_are_checked_even_when_named_like_transport_metadata(
    tiny_release: tuple[Path, dict[str, str]], relative: str
) -> None:
    root, _ = tiny_release
    add_transport_files(root, (relative,))
    source = seal_tiny_release(root, ("meta/pi05_norm_stats.json", "payload.bin", relative))
    assert preflight.verify_release(root, source) == 4
    path = root / relative
    path.write_bytes(b"T" + path.read_bytes()[1:])
    with pytest.raises(ValueError, match="Release content differs"):
        preflight.verify_release(root, source)


def test_transport_metadata_does_not_allow_symlinked_payload(
    tiny_release: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    root, source = tiny_release
    add_transport_files(root, (".cache/huggingface/download/payload.bin.metadata",))
    outside = tmp_path / "outside.bin"
    outside.write_bytes((root / "payload.bin").read_bytes())
    (root / "payload.bin").unlink()
    (root / "payload.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="Missing, symlinked or external release file"):
        preflight.verify_release(root, source)


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_unlisted_checksum_catalog_is_optional(
    tiny_release: tuple[Path, dict[str, str]], change: str
) -> None:
    root, source = tiny_release
    if change == "missing":
        (root / "SHA256SUMS").unlink()
    else:
        (root / "SHA256SUMS").write_text("outdated secondary catalog\n")
    assert preflight.verify_release(root, source) == 3
    # The manifest still detects corruption even without a usable second catalog.
    (root / "payload.bin").write_bytes(b"abd")
    with pytest.raises(ValueError, match="Release content differs"):
        preflight.verify_release(root, source)


def test_manifest_listed_checksum_catalog_is_still_hashed(
    tiny_release: tuple[Path, dict[str, str]],
) -> None:
    root, source = tiny_release
    manifest_path = root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checksums = root / "SHA256SUMS"
    manifest["files"]["SHA256SUMS"] = {
        "bytes": checksums.stat().st_size,
        "sha256": hashlib.sha256(checksums.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest))
    source["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert preflight.verify_release(root, source) == 4
    checksums.write_bytes(b"0" + checksums.read_bytes()[1:])
    with pytest.raises(ValueError, match="Release content differs"):
        preflight.verify_release(root, source)


@pytest.fixture
def launcher(tmp_path: Path):
    """Replace Python with an argv recorder so these tests cannot start Ray or training."""
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps({'argv': sys.argv[1:], "
        "'cwd': os.getcwd(), "
        "'pythonpath': os.environ['PYTHONPATH'], "
        "'offline': os.environ['HF_HUB_OFFLINE']}) + '\\n')\n"
        "if '.preflight' in ' '.join(sys.argv):\n"
        "    sys.exit(int(os.environ.get('PREFLIGHT_EXIT', '0')))\n"
    )
    fake_python.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    config = tmp_path / "run with spaces/train.yaml"
    config.parent.mkdir()
    config.write_text("{}\n")
    extra_dependencies = tmp_path / "extra-dependencies"
    extra_dependencies.mkdir()
    env = {
        **os.environ,
        "PYTHON": str(fake_python),
        "CALL_LOG": str(log),
        "PYTHONPATH": str(extra_dependencies),
    }
    env.pop("RAY_ADDRESS", None)
    env.pop("PREFLIGHT_EXIT", None)

    def invoke(
        *arguments: str,
        ray_address: str | None = None,
        preflight_exit: int = 0,
        script_path: Path = HERE / "run.sh",
    ):
        process_env = {**env, "PREFLIGHT_EXIT": str(preflight_exit)}
        if ray_address is not None:
            process_env["RAY_ADDRESS"] = ray_address
        result = subprocess.run(
            ["/bin/bash", str(script_path), "--config", str(config), *arguments],
            env=process_env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls, config

    return invoke


def test_check_only_needs_no_ray_service(launcher) -> None:
    result, calls, config = launcher("--check-only")
    assert result.returncode == 0, result.stderr
    assert calls == [
        {
            "argv": ["-m", f"{RECIPE}.preflight", "--config", str(config)],
            "cwd": str(REPO_ROOT),
            "pythonpath": f"{REPO_ROOT}/src:{REPO_ROOT}:{config.parents[1]}/extra-dependencies",
            "offline": "1",
        }
    ]
    assert not (config.parent / "model").exists()


@pytest.mark.parametrize("address", [None, "", "auto", "local", "AUTO", " localhost:6379"])
def test_launch_requires_explicit_ray_address(launcher, address: str | None) -> None:
    result, calls, _ = launcher(ray_address=address)
    assert result.returncode == 2
    assert "explicit address" in result.stderr
    assert calls == []


def test_launch_forwards_config_to_preflight_and_official_cli(launcher) -> None:
    result, calls, config = launcher(ray_address="192.0.2.10:6379")
    assert result.returncode == 0, result.stderr
    assert [call["argv"] for call in calls] == [
        ["-m", f"{RECIPE}.preflight", "--config", str(config), "--reserve-output"],
        ["-u", "-m", "carrot.cli.train_sft", "--config", str(config)],
    ]
    assert all(call["cwd"] == str(REPO_ROOT) for call in calls)
    assert all(
        call["pythonpath"].split(os.pathsep)
        == [
            str(REPO_ROOT / "src"),
            str(REPO_ROOT),
            str(config.parents[1] / "extra-dependencies"),
        ]
        for call in calls
    )


@pytest.mark.parametrize("check_only", [False, True])
def test_full_validation_flag_only_reaches_preflight(launcher, check_only: bool) -> None:
    result, calls, config = launcher(
        "--full",
        *(("--check-only",) if check_only else ()),
        ray_address=None if check_only else "192.0.2.10:6379",
    )
    assert result.returncode == 0, result.stderr
    expected = ["-m", f"{RECIPE}.preflight", "--config", str(config), "--full"]
    if not check_only:
        expected.append("--reserve-output")
    assert calls[0]["argv"] == expected
    assert len(calls) == (1 if check_only else 2)
    if not check_only:
        assert calls[1]["argv"] == ["-u", "-m", "carrot.cli.train_sft", "--config", str(config)]


@pytest.mark.parametrize("missing", ["pyproject", "training_cli", "both"])
@pytest.mark.parametrize("check_only", [False, True])
def test_incomplete_checkout_cannot_fall_back_to_installed_carrot(
    launcher,
    tmp_path: Path,
    missing: str,
    check_only: bool,
) -> None:
    orphan_root = tmp_path / "orphan-source"
    orphan_recipe = orphan_root / "recipes" / HERE.name
    orphan_recipe.mkdir(parents=True)
    orphan_script = orphan_recipe / "run.sh"
    orphan_script.write_bytes((HERE / "run.sh").read_bytes())
    (orphan_recipe / "preflight.py").write_bytes((HERE / "preflight.py").read_bytes())
    if missing not in ("pyproject", "both"):
        (orphan_root / "pyproject.toml").write_text('[project]\nname = "carrot-rl"\n')
    if missing not in ("training_cli", "both"):
        source_cli = orphan_root / "src/carrot/cli/train_sft.py"
        source_cli.parent.mkdir(parents=True)
        source_cli.write_text('raise RuntimeError("Incomplete checkout")\n')
    installed_package = tmp_path / "extra-dependencies/carrot"
    (installed_package / "cli").mkdir(parents=True)
    (installed_package / "__init__.py").write_text("")
    (installed_package / "cli/__init__.py").write_text("")
    (installed_package / "cli/train_sft.py").write_text(
        'raise RuntimeError("An installed wheel must not replace the source checkout")\n'
    )
    result, calls, _ = launcher(
        *(("--check-only",) if check_only else ()),
        ray_address="192.0.2.10:6379",
        script_path=orphan_script,
    )
    assert result.returncode == 2
    assert "requires a complete Carrot source checkout" in result.stderr
    assert calls == []


def test_failed_preflight_prevents_training_cli(launcher) -> None:
    result, calls, _ = launcher(ray_address="192.0.2.10:6379", preflight_exit=7)
    assert result.returncode == 7
    assert len(calls) == 1
    assert calls[0]["argv"][1] == f"{RECIPE}.preflight"
    assert "--reserve-output" in calls[0]["argv"]


def test_unknown_launcher_arguments_are_rejected(launcher) -> None:
    result, calls, _ = launcher("--resume", "checkpoint")
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert calls == []
