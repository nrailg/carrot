import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import lerobot
import numpy as np
import torch
from transformers import AutoTokenizer

from common import VENV, WHEEL_SHA256, load_dataset, resources
from configuration_pi05_control import PI05ControlConfig
from modeling_pi05_control import PI05ControlPolicy
from state_jitter import StateJitterStep
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def dataset_joint_names(root: Path) -> list[str]:
    info = json.loads((root / "meta/info.json").read_text())
    names = info["features"]["action"]["names"]
    assert names == [
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
    ]
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    assert Path(sys.prefix) == VENV, sys.prefix
    assert sys.prefix != sys.base_prefix
    assert not any("/opt/venvs/carrot/" in p for p in sys.path), sys.path
    args.run_dir.mkdir(parents=True, exist_ok=False)
    source = args.run_dir / "source"
    shutil.copytree(Path(__file__).parent, source, ignore=shutil.ignore_patterns("__pycache__"))
    dfs = Path(os.environ["MY_DFS"])
    wheel = (
        dfs / "experiments/carrot/lerobot_official_arm_dance_packages"
        / "lerobot-0.6.1-py3-none-any.whl"
    )
    assert sha256(wheel) == WHEEL_SHA256
    site = Path(lerobot.__file__).parent.parent
    audited = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("lerobot/") and name.endswith(".py"):
                expected = hashlib.sha256(archive.read(name)).hexdigest()
                assert sha256(site / name) == expected, name
                audited[name] = expected
    (args.run_dir / "upstream_source_sha256.json").write_text(json.dumps(audited, indent=2))

    root, base, tokenizer = resources()
    assert all(p.is_dir() for p in [root, base, tokenizer])
    initial = Path(os.environ["INITIAL_MODEL_DIR"])
    initial.mkdir()
    for name in ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
        shutil.copy2(base / name, initial / name)
    (initial / "model.safetensors").symlink_to(base / "model.safetensors")
    processors = json.loads((initial / "policy_preprocessor.json").read_text())
    tokenizer_steps = [s for s in processors["steps"] if s["registry_name"] == "tokenizer_processor"]
    assert len(tokenizer_steps) == 1
    assert tokenizer_steps[0]["config"]["tokenizer_name"] == "google/paligemma-3b-pt-224"
    tokenizer_steps[0]["config"]["tokenizer_name"] = str(tokenizer)
    calibration_path = (
        dfs / ".cache/huggingface/lerobot/calibration/robots/so_follower"
        / "my_awesome_follower_arm.json"
    )
    calibration = json.loads(calibration_path.read_text())
    names = [name.removesuffix(".pos") for name in dataset_joint_names(root)]
    # SO101 STS3215: the SDK converts calibrated endpoints with 4095 ticks per turn.
    spans = np.array([
        calibration[name]["range_max"] - calibration[name]["range_min"] for name in names[:5]
    ])
    bounds = np.stack((-spans * 180 / 4095, spans * 180 / 4095))
    bounds32 = bounds.astype(np.float32)
    bounds32[0] = np.where(
        bounds32[0].astype(np.float64) < bounds[0],
        np.nextafter(bounds32[0], np.float32(np.inf)), bounds32[0],
    )
    bounds32[1] = np.where(
        bounds32[1].astype(np.float64) > bounds[1],
        np.nextafter(bounds32[1], np.float32(-np.inf)), bounds32[1],
    )
    jitter = StateJitterStep(lower=bounds32[0].tolist(), upper=bounds32[1].tolist())
    processors["steps"].insert(
        0, {"registry_name": "arm_dance_state_jitter", "config": jitter.get_config()}
    )
    raw = torch.zeros(1024, 6)
    raw[:, 5] = 42
    augmented = jitter.observation({"observation.state": raw})["observation.state"]
    assert torch.equal(augmented[:, 5], raw[:, 5]) and torch.equal(raw[:, :5], torch.zeros(1024, 5))
    assert augmented[:, :5].abs().max() <= 3 and augmented[:, :5].abs().mean() > 1
    disabled = StateJitterStep(**(jitter.get_config() | {"enabled": False}))
    assert torch.equal(disabled.observation({"observation.state": raw})["observation.state"], raw)
    dummy = PI05ControlPolicy.__new__(PI05ControlPolicy)
    torch.nn.Module.__init__(dummy)
    dummy.register_parameter("probe", torch.nn.Parameter(torch.zeros(1)))
    dummy.config = PI05ControlConfig(device="cpu", empty_cameras=2)
    dummy.config.input_features = {
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
    }
    dummy.config.validate_features()
    images, masks = dummy._preprocess_images({"observation.images.wrist": torch.rand(2, 3, 224, 224)})
    assert len(images) == 3 and all(torch.count_nonzero(image) == 0 for image in images)
    assert all(not mask.any() for mask in masks)
    assert PI05ControlPolicy.forward is PI05ControlPolicy.__mro__[1].forward
    assert PI05ControlPolicy.predict_action_chunk is PI05ControlPolicy.__mro__[1].predict_action_chunk
    (initial / "policy_preprocessor.json").write_text(json.dumps(processors, indent=2) + "\n")
    tokens = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    assert tokens("Dance with the arm")["input_ids"]

    dataset = load_dataset(PI05Config(chunk_size=10, n_action_steps=1, device="cpu"))
    assert len(dataset) == 144 and dataset.num_episodes == 1
    for key in ["observation.state", "action"]:
        for stat in ["q01", "q99"]:
            assert np.isfinite(dataset.meta.stats[key][stat]).all()
    for i in [0, 72, 143]:
        sample = dataset[i]
        assert sample["task"] == "Dance with the arm"
        assert sample["action"].shape == (10, 6)
        assert sample["observation.state"].shape == (6,)
        assert torch.isfinite(sample["action"]).all()
        image = sample["observation.images.wrist"]
        assert image.shape == (3, 480, 640) and image.dtype == torch.uint8
        assert np.asarray(image).std() > 0
    features = dataset_to_policy_features(dataset.meta.features)
    cfg = PI05ControlConfig(device="cpu", chunk_size=10, n_action_steps=1, empty_cameras=2)
    cfg.input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    cfg.output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
    cfg.validate_features()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(initial),
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "normalizer_processor": {
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping, "stats": dataset.meta.stats,
            },
        },
        postprocessor_overrides={"unnormalizer_processor": {
            "features": cfg.output_features, "norm_map": cfg.normalization_mapping,
            "stats": dataset.meta.stats,
        }},
    )
    sample = dataset[0]
    processed = preprocessor(sample.copy())
    assert processed["observation.language.tokens"].shape == (1, 200)
    restored = postprocessor(processed["action"])
    assert torch.allclose(restored[0], sample["action"], atol=1e-4, rtol=1e-5)
    files = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()}
    assert files["data/chunk-000/file-000.parquet"] == (
        "752fb901eab1ddb0892ceee2ed4b68afe36fabdb971a805db40a28c3424f8629"
    )
    assert not any(name == "carrot" or name.startswith("carrot.") for name in sys.modules)
    result = {
        "status": "passed",
        "python": sys.version,
        "prefix": sys.prefix,
        "sys_path": sys.path,
        "upstream_python_files": len(audited),
        "wheel_sha256": WHEEL_SHA256,
        "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
        "dataset_sha256": files,
        "base_model_sha256": sha256(base / "model.safetensors"),
        "initial_model_dir": str(initial),
        "processor_override": {"tokenizer_name": str(tokenizer), "state_jitter": jitter.get_config()},
        "vision": False,
        "calibration_sha256": sha256(calibration_path),
        "base_revision": (
            base / ".cache/huggingface/download/config.json.metadata"
        ).read_text().splitlines()[0],
        "tokenizer_sha256": {p.name: sha256(p) for p in tokenizer.iterdir() if p.is_file()},
        "docker_image_tag": "未记录",
        "carrot_commit": os.environ["SOURCE_COMMIT"],
        "upstream_commit": "未记录；官方 PyPI 0.6.1 wheel SHA 已核验",
    }
    (args.run_dir / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PREFLIGHT PASSED", flush=True)


if __name__ == "__main__":
    main()
