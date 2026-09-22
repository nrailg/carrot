import argparse
import json
from pathlib import Path

from extract_fixture import extract_fixture

OBJECTS = {
    "bowl": "Bowl008",
    "plate": "Plate012",
    "ramekin": "Bowl009",
    "cookies": "Cookies002",
    "bottle": "Bottle054",
    "frying_pan": "Pot086",
}
FIXTURES = {
    "cabinet": "/world/storage_furniture_right_group_1",
    "stove": "/world/stovetop_front_group_1",
    "moka_pot": "/world/mokapot_1_front_group_1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--object-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets = {
        name: (args.object_cache / asset / f"{asset}.usd").resolve(strict=True)
        for name, asset in OBJECTS.items()
    }
    scene = args.scene.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "scene.usd").symlink_to(scene)
    for name, path in assets.items():
        (args.output / f"{name}.usd").symlink_to(path)
    metadata = {
        name: extract_fixture(scene, prim, args.output / f"{name}.usd")
        for name, prim in FIXTURES.items()
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
