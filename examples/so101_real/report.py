import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .config import JOINT_NAMES


def write_report(directory: Path) -> None:
    # 只统计已写入 action 事件的动作，忽略预测后尚未消费的行。
    completed = {}
    joint_units = None
    gripper_units = None
    with (directory / "events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "metadata":
                joint_units = event["metadata"]["joint_units"]
                gripper_units = event["metadata"]["gripper_units"]
            if event["event"] == "action":
                completed[event["chunk"]] = completed.get(event["chunk"], 0) + 1
    predictions = []
    references = []
    frames = []
    for path in sorted(directory.glob("chunk_*.npz")):
        with np.load(path) as chunk:
            count = completed.get(int(path.stem.split("_")[1]), 0)
            if count == 0:
                continue
            assert count <= int(chunk["planned_steps"]), "more actions logged than planned"
            predictions.append(chunk["predicted"][:count])
            frames.append(int(chunk["frame"]) + np.arange(count))
            if "reference" in chunk:
                references.append(chunk["reference"][:count])

    if not predictions:
        return
    assert joint_units == "degrees" and gripper_units == "percentage_points", (
        "action reports require policy unit metadata from events.jsonl"
    )
    prediction = np.concatenate(predictions)
    frame_indices = np.concatenate(frames)
    metrics = {"compared_frames": 0}
    reference = np.concatenate(references) if references else None
    if reference is not None:
        assert reference.shape == prediction.shape, "inconsistent reference action coverage"
        metrics = {"compared_frames": len(prediction), "joint_mae": dict(zip(
            JOINT_NAMES, np.abs(prediction - reference).mean(axis=0).astype(float), strict=True
        ))}
    metrics["joint_units"] = joint_units
    metrics["gripper_units"] = gripper_units
    (directory / "comparison.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(frame_indices, prediction[:, index], label="prediction", marker=".")
        if reference is not None:
            axis.plot(frame_indices, reference[:, index], label="demonstration", marker=".")
        axis.set_title(JOINT_NAMES[index])
        axis.set_xlabel("episode frame")
        axis.set_ylabel(gripper_units if index == 5 else joint_units)
        axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "actions.png")
    plt.close(figure)
