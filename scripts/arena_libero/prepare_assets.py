import argparse
import json
from pathlib import Path

from extract_fixture import extract_fixture

from carrot_sim.arena_libero.tasks.assets import ASSETS

FIXTURES = {
    "cabinet": "/world/storage_furniture_right_group_1",
    "stove": "/world/stovetop_front_group_1",
    "moka_pot": "/world/mokapot_1_front_group_1",
    "microwave": "/world/microwave_axis_group_1",
    "wine_rack": "/world/winerack_left_group_1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--object-cache", type=Path, required=True)
    parser.add_argument("--libero-bbq-usd", type=Path, required=True)
    parser.add_argument("--shelf-usd", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets = {
        name: args.libero_bbq_usd.resolve(strict=True)
        if name == "bbq_sauce"
        else (args.object_cache / geometry.asset_id / f"{geometry.asset_id}.usd").resolve(
            strict=True
        )
        for name, geometry in ASSETS.items()
        if name not in FIXTURES
    }
    scene = args.scene.resolve(strict=True)
    if args.shelf_usd is not None:
        assets["shelf"] = args.shelf_usd.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "scene.usd").symlink_to(scene)
    for name, path in assets.items():
        (args.output / f"{name}.usd").symlink_to(path)
    metadata = {
        name: extract_fixture(
            scene, prim, args.output / f"{name}.usd", static_body=name == "wine_rack"
        )
        for name, prim in FIXTURES.items()
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
