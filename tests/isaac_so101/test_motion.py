import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_so101_sweeps_with_fixed_base(tmp_path: Path) -> None:
    # 用真实 PhysX 跑两个周期，防止只有目标在变、机器人不动或整个基座一起转。
    # 资产必须由调用者提供；不下载、不跳过，也不以 mock 代替物理仿真。
    asset = Path(os.environ["SO101_USD"])
    assert asset.is_file(), asset
    script = Path(__file__).resolve().parents[2] / "examples/isaac_so101/run_isaac_sim.py"
    trace = tmp_path / "motion.npz"

    # 有限运行与交互演示使用同一个循环；保留完整启动日志以定位资产或驱动问题。
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--robot-usd",
            str(asset),
            "--headless",
            "--gpu",
            os.environ.get("SO101_GPU", "0"),
            "--steps",
            "960",
            "--trace",
            str(trace),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"simulator exited with {result.returncode}"

    # 读取物理实测值，第二周期排除初始瞬态；关节顺序由模型报告而非硬编码下标。
    with np.load(trace, allow_pickle=False) as data:
        names = data["joint_names"].tolist()
        positions, targets = data["positions"], data["targets"]
        base_positions, base_rotations = data["base_positions"], data["base_rotations"]
    assert positions.shape == targets.shape == (960, 6)
    assert all(np.isfinite(a).all() for a in (positions, targets, base_positions, base_rotations))
    pan = names.index("shoulder_pan")
    moving = positions[480:, pan]
    error = moving - targets[480:, pan]

    # 实际左右均达到至少约23度；位置误差允许 PD 的相位滞后，但拒绝卡住或失控。
    assert moving.max() > 0.4 and moving.min() < -0.4, (moving.min(), moving.max())
    assert np.sqrt(np.mean(error**2)) < 0.15, f"pan tracking RMSE: {np.sqrt(np.mean(error**2))}"
    assert np.abs(positions[:, pan]).max() < np.deg2rad(35)

    # 其余关节须保持在零位附近，基座平移小于1毫米、旋转不变（允许四元数正负等价）。
    held = [i for i in range(6) if i != pan]
    assert np.abs(positions[480:, held]).max() < 0.05
    assert np.linalg.norm(base_positions - base_positions[0], axis=1).max() < 0.001
    assert np.abs(base_rotations @ base_rotations[0]).min() > 0.99999
