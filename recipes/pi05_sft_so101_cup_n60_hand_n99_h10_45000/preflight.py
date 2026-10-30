"""Check inputs, optionally reserve a fresh output directory; never start Ray or a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from carrot.data.loading import load_callable

HERE = Path(__file__).resolve().parent
FACTORY = f"recipes.{HERE.name}.dataset.build_dataset"
NORM_SHA256 = "ed6087fd3e59c7e103a295abe2811241d7b30f5ae88975258d348e4a747dac0b"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def release_file(root: Path, relative: str) -> Path:
    """Resolve only canonical regular files inside the selected release."""
    posix = PurePosixPath(relative)
    require(
        bool(relative)
        and not posix.is_absolute()
        and str(posix) == relative
        and ".." not in posix.parts
        and "\\" not in relative,
        f"Invalid release path: {relative!r}",
    )
    path = root / relative
    require(
        path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
        f"Missing, symlinked or external release file: {path}",
    )
    return path


def is_hub_metadata(relative: Path) -> bool:
    """Identify repository-root Git metadata and the HF local-download cache."""
    return (
        relative.parts == (".gitattributes",)
        or relative.parts[:1] == (".git",)
        or relative.parts[:2] == (".cache", "huggingface")
    )


def verify_release(root: Path, source: dict[str, Any]) -> int:
    """Bind local bytes to the pinned release, rather than trusting its directory name."""
    root = root.resolve()
    manifest_path = release_file(root, "release_manifest.json")
    require(sha256(manifest_path) == source["manifest_sha256"], f"Release manifest differs: {root}")
    manifest = read_json(manifest_path)
    require(manifest["dataset_id"] == source["repo_id"], f"Dataset identity differs: {root}")
    expected = set(manifest["files"]) | {"release_manifest.json"}
    observed = (path.relative_to(root) for path in root.rglob("*") if path.is_file())
    # A manifest entry remains mandatory even if its path matches a metadata exclusion.
    actual = {
        path.as_posix()
        for path in observed
        if path.as_posix() in expected
        or not (
            is_hub_metadata(path) or path.name == ".DS_Store" or path.as_posix() == "SHA256SUMS"
        )
    }
    require(actual == expected, f"Unexpected or missing release files: {root}")
    # The pinned manifest is the single content inventory. An unlisted SHA256SUMS
    # is an optional release sidecar, not a second source of truth.
    for relative, item in manifest["files"].items():
        path = release_file(root, relative)
        require(
            path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"],
            f"Release content differs: {path}",
        )
    return len(expected)


def require_integer(value: Any, label: str, minimum: int = 0) -> None:
    require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")


def check_training_parameters(config: dict[str, Any]) -> None:
    """Accept tunable training settings while rejecting invalid sizes and optimizer values."""
    for key in (
        "steps",
        "micro_batch_size",
        "global_batch_size",
        "log_freq",
        "dp_size",
        "num_nodes",
    ):
        require_integer(config[key], key, 1)
    require_integer(config["save_freq"], "save_freq")
    require(type(config["seed"]) is int, "seed must be an integer")
    require_integer(config["dataset"]["num_workers"], "dataset.num_workers")
    require(config["dp_size"] % config["num_nodes"] == 0, "dp_size must be divisible by num_nodes")
    require(
        config["global_batch_size"] % (config["micro_batch_size"] * config["dp_size"]) == 0,
        "global_batch_size must be divisible by micro_batch_size * dp_size",
    )
    # Carrot has no DDP fallback when FSDP is disabled.
    require(
        config["dp_size"] == 1 or config["fsdp"]["enabled"] is True,
        "Multi-GPU training requires fsdp.enabled=true to synchronize gradients",
    )
    optimizer = config["optimizer"]
    for key in ("learning_rate", "decay_learning_rate", "eps", "max_grad_norm", "weight_decay"):
        value = optimizer[key]
        bound = "nonnegative" if key == "weight_decay" else "positive"
        require(
            type(value) in (int, float)
            and np.isfinite(value)
            and (value >= 0 if key == "weight_decay" else value > 0),
            f"optimizer.{key} must be finite and {bound}",
        )
    require(
        optimizer["decay_learning_rate"] <= optimizer["learning_rate"],
        "optimizer.decay_learning_rate cannot exceed learning_rate",
    )
    require_integer(optimizer["warmup_steps"], "optimizer.warmup_steps")
    require_integer(optimizer["decay_steps"], "optimizer.decay_steps", 1)
    betas = optimizer["betas"]
    require(
        isinstance(betas, (list, tuple))
        and len(betas) == 2
        and all(type(beta) in (int, float) and 0 <= beta < 1 for beta in betas),
        "optimizer.betas must contain two values in [0, 1)",
    )


def check_output_path(config: dict[str, Any]) -> Path:
    """Resolve a fresh output path outside the checkout and all configured inputs."""
    output = Path(config["output_dir"])
    require(output.is_absolute(), "Fresh training requires an absolute output directory")
    require(
        not output.exists() and not output.is_symlink(),
        f"Output directory already exists or is reserved: {output}. Use a new run directory; "
        "existing outputs are never removed automatically.",
    )
    output = output.resolve()
    protected = [
        HERE.parents[1],
        *(Path(item["root"]) for item in config["dataset"]["factory_kwargs"]["datasets"]),
        Path(config["model"]["path"]),
        Path(config["model"]["tokenizer_path"]),
    ]
    for root in protected:
        require(
            not output.is_relative_to(root.resolve()),
            f"Output directory must be outside the source and input directories: {root}",
        )
    return output


def reserve_output_directory(config: dict[str, Any]) -> Path:
    """Atomically claim a fresh run after validation; never reuse or remove an output."""
    output = check_output_path(config)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Output directory was already reserved: {output}. Use a new run directory."
        ) from error
    return output


def check_config(config: dict[str, Any]) -> tuple[list[dict[str, Any]], Path]:
    """Check input compatibility and valid settings without locking template hyperparameters."""
    check_training_parameters(config)
    dataset = config["dataset"]
    require(dataset["factory"] == FACTORY, "Use this recipe's local dataset factory")
    require(dataset["norm_stats_source"] == "file", "Use the saved joint normalization file")
    require(dataset["preprocess"] is None, "SO101 samples must not use an extra preprocessor")
    kwargs = dataset["factory_kwargs"]
    require(kwargs["action_horizon"] == 10, "This recipe requires H10 action windows")
    require(
        (kwargs["base_image_key"], kwargs["wrist_image_key"])
        == ("observation.images.top", "observation.images.wrist"),
        "Expected top and wrist cameras in their original order",
    )
    sources = read_json(HERE / "inputs.json")["sources"]
    require(len(kwargs["datasets"]) == len(sources), "Expected the two pinned source releases")
    for supplied, expected in zip(kwargs["datasets"], sources, strict=True):
        for key in ("repo_id", "revision", "frames", "episodes", "manifest_sha256"):
            require(supplied[key] == expected[key], f"Source release parameter differs: {key}")
        require(Path(supplied["root"]).is_absolute(), "Dataset roots must be absolute")
    norm = Path(dataset["norm_stats_path"])
    require(norm.is_absolute() and norm.is_file(), "Saved absolute normalization path is required")
    require(
        Path(kwargs["norm_stats_path"]).resolve() == norm.resolve(),
        "Dataset and model must use the same joint normalization",
    )
    require(sha256(norm) == NORM_SHA256, "Joint normalization differs from the original experiment")
    for key in ("path", "tokenizer_path"):
        require(Path(config["model"][key]).is_absolute(), f"Model {key} must be absolute")
    check_output_path(config)
    return sources, norm


def check_samples(config: dict[str, Any], norm: Path, *, full: bool = False) -> dict[str, Any]:
    """Decode three anchors per source, or audit every episode and statistics with ``full``."""
    kwargs = config["dataset"]["factory_kwargs"]
    spec = load_callable(config["dataset"]["factory"])(**kwargs)
    horizon = kwargs["action_horizon"]
    state_parts, action_parts = [], []
    cursor = samples = episodes_checked = 0
    for source in kwargs["datasets"]:
        root = Path(source["root"])
        if full:
            rows = [
                row
                for path in sorted((root / "data").rglob("*.parquet"))
                for row in pq.read_table(path).to_pylist()
            ]
            rows.sort(key=lambda row: row["index"])
            require(
                [row["index"] for row in rows] == list(range(source["frames"])),
                f"Noncontiguous source frame indices: {root}",
            )
            states = np.asarray([row["observation.state"] for row in rows], dtype=np.float64)
            actions = np.asarray([row["action"] for row in rows], dtype=np.float64)
            require(
                states.shape == actions.shape == (source["frames"], 6)
                and np.isfinite(states).all()
                and np.isfinite(actions).all(),
                f"Invalid state or action vectors: {root}",
            )
            state_parts.append(states)
            action_parts.append(actions)
            tasks = {
                row["task_index"]: row["task"]
                for row in pq.read_table(root / "meta/tasks.parquet").to_pylist()
            }
        features = read_json(root / "meta/info.json")["features"]
        anchors = {0, source["frames"] // 2, source["frames"] - 1}
        episodes = [
            row
            for path in sorted((root / "meta/episodes").rglob("*.parquet"))
            for row in pq.read_table(
                path,
                columns=[
                    "episode_index",
                    "dataset_from_index",
                    "dataset_to_index",
                    "length",
                    "tasks",
                ],
            ).to_pylist()
        ]
        episodes.sort(key=lambda row: row["episode_index"])
        require(len(episodes) == source["episodes"], f"Episode count differs: {root}")
        end_previous = 0
        for episode in episodes:
            start, end = episode["dataset_from_index"], episode["dataset_to_index"]
            require(
                start == end_previous
                and end > start
                and end <= source["frames"]
                and end - start == episode["length"],
                f"Invalid episode boundaries: {root}",
            )
            selected = {start, (start + end) // 2, end - 1} if full else anchors
            for index in sorted(index for index in selected if start <= index < end):
                sample = spec.dataset[cursor + index]
                for key, shape in (("observation/state", (6,)), ("actions", (horizon, 6))):
                    values = np.asarray(sample[key])
                    require(
                        values.shape == shape
                        and values.dtype == np.float32
                        and np.isfinite(values).all(),
                        f"Invalid adapter {key} shape/dtype/values at {root}:{index}",
                    )
                if full:
                    indices = np.minimum(np.arange(index, index + horizon), end - 1)
                    require(
                        np.array_equal(sample["observation/state"], states[index]),
                        f"Adapter state differs at {root}:{index}",
                    )
                    require(
                        np.array_equal(sample["actions"], actions[indices]),
                        f"Adapter action window crosses an episode or differs at {root}:{index}",
                    )
                    require(
                        sample["prompt"] == tasks[rows[index]["task_index"]],
                        f"Adapter task differs at {root}:{index}",
                    )
                require(
                    np.array_equal(
                        sample["action_is_pad"], np.arange(index, index + horizon) >= end
                    ),
                    f"Adapter padding differs at {root}:{index}",
                )
                require(
                    sample["prompt"] in episode["tasks"],
                    f"Adapter task differs at {root}:{index}",
                )
                for key, camera in (
                    ("observation/image", kwargs["base_image_key"]),
                    ("observation/wrist_image", kwargs["wrist_image_key"]),
                ):
                    pixels = np.asarray(sample[key])
                    height, width, channels = features[camera]["shape"]
                    require(
                        pixels.dtype == np.uint8 and pixels.shape == (channels, height, width),
                        f"Decoded camera shape/dtype differs at {root}:{index}:{key}",
                    )
                samples += 1
            end_previous = end
            episodes_checked += 1
        require(end_previous == source["frames"], f"Episode coverage differs: {root}")
        cursor += source["frames"]
    require(len(spec.dataset) == cursor, "Mixed dataset length differs")
    if full:
        statistics = read_json(norm)
        for key, parts in (("state", state_parts), ("actions", action_parts)):
            values = np.concatenate(parts)
            expected = {
                "mean": values.mean(0),
                "std": values.std(0),
                "q01": np.quantile(values, 0.01, axis=0),
                "q99": np.quantile(values, 0.99, axis=0),
            }
            for statistic, value in expected.items():
                require(
                    np.allclose(statistics[key][statistic], value, rtol=0, atol=1e-10),
                    f"Joint statistics differ from source frame vectors: {key}.{statistic}",
                )
    return {
        "frames": cursor,
        "episodes": episodes_checked,
        "samples_checked": samples,
        "statistics_recomputed": full,
    }


def validate(
    config_path: Path,
    *,
    check_model: bool = True,
    full: bool = False,
    reserve_output: bool = False,
) -> dict[str, Any]:
    """Validate local data and model configuration, without model load or GPU work.

    ``check_model=False`` permits local data-only validation without GPU model
    assets. ``full=True`` also audits every episode's windows and recomputes
    normalization for comparison. Neither mode downloads files, starts Ray,
    or changes any inputs. Both modes hash every manifest-listed release file.
    ``reserve_output=True`` is for the training launcher: after all checks pass,
    atomically create the output directory. Other calls remain read-only.
    """
    require(not reserve_output or check_model, "Output reservation requires model preflight")
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    sources, norm = check_config(config)
    if check_model:
        model = Path(config["model"]["path"])
        tokenizer = Path(config["model"]["tokenizer_path"])
        require(model.is_dir() and tokenizer.is_dir(), "Local model/tokenizer directories required")
        model_config = read_json(model / "config.json")
        require(
            model_config["action_horizon"] == 10 and model_config["action_dim"] == 32,
            "Expected the original H10 PI0.5 base configuration with action_dim=32",
        )
    files = sum(
        verify_release(Path(item["root"]), source)
        for item, source in zip(
            config["dataset"]["factory_kwargs"]["datasets"], sources, strict=True
        )
    )
    report = {
        "status": "passed",
        "mode": "preflight" if check_model else "data-only",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "norm_sha256": sha256(norm),
        "release_files_checked": files,
        "full_validation": full,
        **check_samples(config, norm, full=full),
        "model_config_checked": check_model,
        "model_weights_loaded": False,
        "gpu_forward_checked": False,
        "ray_started": False,
        "training_started": False,
        "output_reserved": False,
    }
    if reserve_output:
        require(
            sha256(config_path) == report["config_sha256"],
            "Configuration changed during preflight; validate it again before starting",
        )
        reserve_output_directory(config)
        report["output_reserved"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Check real training data without requiring local model assets",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Audit every episode's beginning/middle/end and recompute joint statistics",
    )
    parser.add_argument(
        "--reserve-output",
        action="store_true",
        help="After all checks pass, atomically claim the output directory for fresh training",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                args.config,
                check_model=not args.data_only,
                full=args.full,
                reserve_output=args.reserve_output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
