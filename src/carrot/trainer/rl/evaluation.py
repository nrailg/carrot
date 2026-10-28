import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from carrot.distributed.group import WorkerGroup
from carrot.trainer.rl.config import TransferConfig

LABELS = ("init", "sft", "rl")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def paired_summary(rows: list[dict]) -> dict:
    """Compare only episode keys completed for every checkpoint, retaining incomplete accounting.

    Parameters
    ----------
    rows : list[dict]
        One complete row per (checkpoint, suite, task, episode); duplicates fail validation.

    Returns
    -------
    dict
        Paired rates and deltas, with Long separated from the three primary suites.
    """
    keyed: dict[tuple, dict] = {}
    for row in rows:
        assert row["checkpoint"] in LABELS and row["suite"] in SUITES
        key = (row["suite"], row["task"], row["episode"])
        group = keyed.setdefault(key, {})
        assert row["checkpoint"] not in group, f"duplicate evaluation result: {key}"
        group[row["checkpoint"]] = row
    paired = {key: group for key, group in keyed.items() if set(group) == set(LABELS)}
    for key, group in paired.items():
        assert len({r["initial_state_sha256"] for r in group.values()}) == 1, key
        assert len({r["seed"] for r in group.values()}) == 1, key
    summary = {
        "completed_episodes": len(rows),
        "paired_cases": len(paired),
        "unpaired_cases": len(keyed) - len(paired),
        "groups": {},
    }
    for name, suites in [
        ("primary", SUITES[:3]),
        ("long", SUITES[3:]),
        *[(suite, (suite,)) for suite in SUITES],
    ]:
        cases = [group for key, group in paired.items() if key[0] in suites]
        counts = {label: sum(bool(group[label]["success"]) for group in cases) for label in LABELS}
        rates = {label: counts[label] / len(cases) if cases else None for label in LABELS}
        changes = {
            f"{label}_minus_init": rates[label] - rates["init"] if cases else None
            for label in ("sft", "rl")
        }
        changes["rl_minus_sft"] = rates["rl"] - rates["sft"] if cases else None
        summary["groups"][name] = {
            "cases": len(cases),
            "successes": counts,
            "rates": rates,
            "paired_deltas": changes,
        }
    return summary


def source_evaluation(
    simulator: WorkerGroup,
    learner: WorkerGroup,
    config: TransferConfig,
    label: str,
    deadline: float,
) -> list[dict]:
    directory = Path(config.output) / "source-eval" / label
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for start in range(0, config.source_eval_episodes, config.num_envs):
        if time.time() >= deadline:
            break
        seeds = list(range(1_000_000 + start, 1_000_000 + start + config.num_envs))
        observations = simulator.call("request", "reset", seeds=seeds).wait()[0]
        active = set(range(config.num_envs))
        frames = []
        chunk = 0
        while active and time.time() < deadline:
            actions = learner.call("infer", observations, config.seed + start + chunk).wait()[0]
            result = simulator.call(
                "request", "step", actions=actions[:, : config.execute_steps], gamma=config.gamma
            ).wait()[0]
            observations = result["observations"]
            if 0 in active:
                frames.append(observations[0]["observation/image"])
            done = np.flatnonzero(result["terminated"] | result["truncated"]).tolist()
            for i in done:
                if i in active:
                    rows.append({"checkpoint": label, **result["info"][i]})
                    active.remove(i)
            if done:
                # Finished slots remain physically independent, and subsequent episodes are ignored.
                observations = simulator.call(
                    "request", "reset", ids=done, seeds=[2_000_000 + start + i for i in done]
                ).wait()[0]
            chunk += 1
        if frames:
            imageio.mimwrite(directory / f"episode-{start}.mp4", frames, fps=4)
        simulator.call(
            "request",
            "snapshot",
            output=str(Path(config.simulator_output) / "source-eval" / label / str(start)),
        ).wait()
        (directory / "episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def target_evaluation(
    simulator: WorkerGroup,
    learner: WorkerGroup,
    config: TransferConfig,
    checkpoints: dict[str, str],
    deadline: float,
) -> dict:
    directory = Path(config.output) / "target-eval"
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for episode in range(config.target_eval_rounds):
        for label in LABELS:
            learner.call("load", checkpoints[label], "libero").wait()
            for suite_id, suite in enumerate(SUITES):
                for task in range(10):
                    if time.time() >= deadline:
                        summary = paired_summary(rows)
                        (directory / "summary.json").write_text(
                            json.dumps(summary, indent=2) + "\n"
                        )
                        return summary
                    seed = config.seed + episode * 40 + suite_id * 10 + task
                    result = simulator.call(
                        "request", "reset", suite=suite, task_id=task, episode=episode, seed=seed
                    ).wait()[0]
                    observations = result["observations"]
                    frames = []
                    chunk = 0
                    while time.time() < deadline:
                        actions = learner.call("infer", observations, seed * 10_000 + chunk).wait()[
                            0
                        ]
                        result = simulator.call(
                            "request", "step", actions=actions[:, : config.execute_steps]
                        ).wait()[0]
                        observations = result["observations"]
                        if task == 0 and episode == 0:
                            frames.append(observations[0]["observation/image"])
                        chunk += 1
                        if result["done"]:
                            row = {
                                "checkpoint": label,
                                "path": checkpoints[label],
                                **result["info"],
                            }
                            rows.append(row)
                            with (directory / "episodes.jsonl").open("a") as stream:
                                stream.write(json.dumps(row) + "\n")
                            print(json.dumps({"target_evaluation": row}), flush=True)
                            break
                    if frames:
                        imageio.mimwrite(
                            directory / f"{label}-{suite}-{task}-{episode}.mp4", frames, fps=4
                        )
        summary = paired_summary(rows)
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return paired_summary(rows)
