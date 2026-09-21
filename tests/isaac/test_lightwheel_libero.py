"""检查真实 Lightwheel 示例的输出，不代替仿真运行。"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageStat


def validate(output: Path, steps: int) -> None:
    # 验证真实 rollout 已走完目标步数，且结果来自指定任务和静止策略。
    result = json.loads((output / "result.json").read_text())
    assert result["task"] == "L90K1PutTheBlackBowlOnThePlate"
    assert result["robot"] == "Panda"
    assert result["policy"] == "hold_current_pose"
    assert result["steps"] == steps
    assert result["num_envs"] == 1
    assert len(result["success_seen"]) == 1
    assert isinstance(result["success_seen"][0], bool)
    assert result["camera_enabled"]

    # 检查落盘 RGB 的尺寸和变化，避免把空白渲染误判为通过。
    with Image.open(output / "camera.png") as image:
        assert image.size == (224, 224)
        assert image.mode == "RGB"
        assert max(ImageStat.Stat(image).stddev) > 1.0
    print("PASS: Lightwheel LIBERO rollout and RGB artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    validate(args.output, args.steps)
