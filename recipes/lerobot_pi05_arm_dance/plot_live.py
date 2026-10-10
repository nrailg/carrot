import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in
            (args.root / "client_retry/trajectory.jsonl").read_text().splitlines()]
    demo = json.loads((args.root / "demonstration.json").read_text())
    assert len(rows) == 144
    prediction = np.array([row["prediction"] for row in rows])
    actual = np.array([row["actual"] for row in rows])
    target = np.array(demo["actions"])
    assert prediction.shape == actual.shape == target.shape == (144, 6)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
    for index, ax in enumerate(axes.flat):
        ax.plot(target[:, index], color="#333333", label="Demonstration", linewidth=1.5)
        ax.plot(prediction[:, index], color="#e78b30", label="Policy output", alpha=0.8,
                linewidth=1)
        ax.plot(actual[:, index], color="#2878b5", label="Measured robot", linewidth=1)
        ax.set_title(demo["action_names"][index].removesuffix(".pos"))
        ax.set_ylabel("degrees" if index < 5 else "recorded gripper units")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Closed-loop step (same-index comparison)")
    fig.suptitle("Official LeRobot PI0.5 / Arm Dance — 144 real-robot steps", fontsize=14)
    fig.savefig(args.root / "trajectory.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
