import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def check_physics(output: Path, num_envs: int, num_colors: int) -> None:
    # 只接受真实物理 runner 的成功记录，图像还需非黑帧且确有多色笔迹。
    result = json.loads((output / "result.json").read_text())
    reset = json.loads((output / "partial-reset.json").read_text())
    assert len(result) == num_envs and reset["partial_reset"] == "PASS"
    # 每个环境都必须完成全部抓取与归还，没有掉笔，且达到像素验收阈值。
    for key, row in result.items():
        assert row["success"] and row["iou"] >= 0.72 and row["drops"] == 0
        assert row["picks"] == row["returns"] == num_colors and row["contact_samples"] > 0
        pixels = np.asarray(Image.open(output / f"env-{key}-canvas.png").convert("RGB"))
        colors = np.unique(pixels.reshape(-1, 3), axis=0)
        assert len(colors) >= num_colors + 1
    # 两个相机都必须提供有效图像，防止渲染异常被训练流程当成正常观测。
    for index in range(num_envs):
        for camera in ("base", "wrist"):
            frame = np.asarray(Image.open(output / "final" / f"env-{index}-{camera}.png"))
            assert frame.shape == (224, 224, 3) and frame.std() > 5
    print(json.dumps({"status": "PASS", "envs": num_envs, "colors": num_colors}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--num-colors", type=int, default=2)
    args = parser.parse_args()
    check_physics(args.output, args.num_envs, args.num_colors)
