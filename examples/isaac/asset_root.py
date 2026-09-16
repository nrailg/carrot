import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault(
    "ISAACSIM_ASSET_ROOT",
    "/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/isaacsim_assets/Assets/Isaac/6.0",
)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import isaaclab.utils.assets as assets


def configure_asset_root_from_env() -> None:
    root = os.environ.get("ISAACSIM_ASSET_ROOT")
    if root is None:
        return

    root_path = Path(root).expanduser().resolve()
    missing = [name for name in ("Isaac", "NVIDIA") if not (root_path / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"ISAACSIM_ASSET_ROOT={root_path} 缺少目录: {', '.join(missing)}"
        )

    root = str(root_path)
    assets.NUCLEUS_ASSET_ROOT_DIR = root
    assets.NVIDIA_NUCLEUS_DIR = f"{root}/NVIDIA"
    assets.ISAAC_NUCLEUS_DIR = f"{root}/Isaac"
    assets.ISAACLAB_NUCLEUS_DIR = f"{root}/Isaac/IsaacLab"
