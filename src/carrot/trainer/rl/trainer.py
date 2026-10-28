import hashlib
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

from carrot.distributed import Cluster, PlacementSpec, RolePlacement, WorkerGroup
from carrot.trainer.rl.config import TransferConfig, save_config
from carrot.trainer.rl.data import verify_local_demonstrations
from carrot.trainer.rl.evaluation import LABELS, source_evaluation, target_evaluation
from carrot.trainer.rl.learner import LearnerWorker
from carrot.trainer.rl.simulator import SimulatorWorker


def copy_artifacts(simulator: WorkerGroup, subdirectory: str, destination: Path) -> dict:
    """Copy simulator artifacts across resource pools without assuming a shared filesystem.

    Parameters
    ----------
    simulator : WorkerGroup
    subdirectory : str
        Relative to the renderer's configured output directory.
    destination : Path
        Fresh directory on the learner/controller's persistent filesystem.

    Returns
    -------
    dict
        File count and verified uncompressed byte count.
    """
    manifest = simulator.call("files", subdirectory).wait()[0]
    assert manifest, f"empty artifact directory: {subdirectory}"
    destination.mkdir(parents=True, exist_ok=False)
    total = 0
    for entry in manifest:
        target = (destination / entry["path"]).resolve()
        assert target.is_relative_to(destination.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        digest = hashlib.sha256()
        offset = 0
        with temporary.open("xb") as stream:
            while offset < entry["size"]:
                packed = simulator.call("read_part", subdirectory, entry["path"], offset).wait()[0]
                block = zlib.decompress(packed)
                assert 0 < len(block) <= 16 * 1024**2
                stream.write(block)
                digest.update(block)
                offset += len(block)
        assert offset == entry["size"] and digest.hexdigest() == entry["sha256"], entry
        os.replace(temporary, target)
        total += offset
    (destination / "transfer-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"files": len(manifest), "bytes": total}


def _event(output: Path, stage: str, **data: object) -> None:
    record = {"time": time.time(), "stage": stage, **data}
    with (output / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def _launch(
    config: TransferConfig,
    cluster: Cluster,
    simulator_num_envs: int | None = None,
) -> tuple[WorkerGroup, WorkerGroup]:
    nodes = tuple(sorted({config.simulator_node_ip, config.learner_node_ip}))
    cluster.reserve(
        "transfer",
        PlacementSpec(
            num_nodes=config.num_nodes,
            bundles_per_node=2 // config.num_nodes,
            cpus_per_bundle=8,
            gpus_per_bundle=1,
            node_ips=nodes,
        ),
    )
    sim_rank = (
        nodes.index(config.simulator_node_ip) if config.num_nodes == 2 else config.simulator_bundle
    )
    model_rank = (
        nodes.index(config.learner_node_ip) if config.num_nodes == 2 else config.learner_bundle
    )
    simulator = cluster.launch(
        "simulator",
        SimulatorWorker,
        config.simulator_script,
        {
            "robot_usd": config.robot_usd,
            "num_envs": config.num_envs if simulator_num_envs is None else simulator_num_envs,
            "max_episode_steps": config.max_episode_steps,
            "seed": config.seed,
            "num_colors": config.num_colors,
        },
        config.simulator_output,
        config.simulator_node_ip,
        placement=RolePlacement("transfer", (sim_rank,), cpus_per_actor=8, gpus_per_actor=1),
        env_vars={"PYTHONPATH": config.simulator_source + "/src", "HF_HUB_OFFLINE": "1"},
    )
    learner = cluster.launch(
        "learner",
        LearnerWorker,
        config,
        placement=RolePlacement("transfer", (model_rank,), cpus_per_actor=8, gpus_per_actor=1),
    )
    return simulator, learner


def run(config: TransferConfig, resume: str | None = None) -> dict:
    """Run demonstration SFT, tracing PPO and paired source/target evaluation within a pilot budget.

    Parameters
    ----------
    config : TransferConfig
        Explicit learner and renderer node addresses and local paths on each pool.
    resume : str | None
        Trusted learner checkpoint; physics restarts with fresh training seeds.

    Returns
    -------
    dict
        Budget status and paired target metrics, without interpreting incomplete cases as failures.
    """
    assert os.environ.get("RAY_ADDRESS"), "start the explicitly placed Ray cluster first"
    started = time.time()
    deadline = started + config.total_hours * 3600
    prepare_deadline = started + config.prepare_hours * 3600
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    save_config(config, output / "config.yaml")
    source_root = Path(__file__).resolve().parents[4]
    files = [
        *source_root.glob("src/carrot/**/*.py"),
        *source_root.glob("examples/isaac_workbench/*.py"),
        *source_root.glob("recipes/pi05_multicolor_transfer/*.yaml"),
        *source_root.glob("recipes/pi05_multicolor_transfer/*.sh"),
    ]
    (output / "source-manifest.json").write_text(
        json.dumps(
            {
                str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files
            },
            indent=2,
        )
        + "\n"
    )
    _event(
        output,
        "start",
        source_revision=os.environ.get("CARROT_SOURCE_REVISION", "not_recorded"),
        image=os.environ.get("CARROT_IMAGE_TAG", "not_recorded"),
        renderer_node=config.simulator_node_ip,
        learner_node=config.learner_node_ip,
        micro_batch_size=config.micro_batch_size,
        global_batch_size=config.global_batch_size,
        global_rollout_batch_size=config.global_rollout_batch_size,
        gradient_accumulation_steps=config.gas,
        rollout_steps=config.rollout_steps,
    )
    checkpoints = {
        "init": str(output / "checkpoints/init"),
        "sft": str(output / "checkpoints/sft"),
        "rl": str(output / "checkpoints/rl"),
    }
    try:
        with Cluster(
            namespace="carrot-multicolor",
            env_vars={
                "PYTHONPATH": os.environ["PYTHONPATH"],
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
        ) as cluster:
            simulator, learner = _launch(
                config, cluster, config.demo_num_envs if resume is None else config.num_envs
            )
            _event(output, "checkpoint_init", **learner.call("save", checkpoints["init"]).wait()[0])
            _event(output, "partial_reset", **simulator.call("request", "check_reset").wait()[0])
            if resume is None:
                if config.learner_demonstrations is not None:
                    dataset = config.learner_demonstrations
                    _event(
                        output,
                        "dataset_reuse",
                        **verify_local_demonstrations(dataset, config.demo_successes),
                    )
                else:
                    collected = simulator.call(
                        "request",
                        "collect",
                        output=str(Path(config.simulator_output) / "demos"),
                        successes=config.demo_successes,
                        attempts=config.demo_attempts,
                        deadline=prepare_deadline,
                        source=config.demonstration_source,
                    ).wait()[0]
                    _event(output, "demonstrations", **collected)
                    assert collected["saved"] == config.demo_successes, (
                        "incomplete demonstrations; inspect physical grasp/draw/return evidence"
                    )
                    _event(
                        output,
                        "dataset_transfer",
                        **copy_artifacts(simulator, "demos", output / "demos"),
                    )
                    dataset = str(output / "demos")
                result = learner.call("sft", dataset, prepare_deadline).wait()[0]
                assert result["sft_updates"] > 0, "preparation budget exhausted before SFT"
                _event(output, "sft", **result)
                _event(
                    output, "checkpoint_sft", **learner.call("save", checkpoints["sft"]).wait()[0]
                )
                transitions = 0
                if config.demo_num_envs != config.num_envs:
                    simulator.call(
                        "restart",
                        config.simulator_script,
                        {
                            "robot_usd": config.robot_usd,
                            "num_envs": config.num_envs,
                            "max_episode_steps": config.max_episode_steps,
                            "seed": config.seed,
                            "num_colors": config.num_colors,
                        },
                    ).wait()
                    _event(
                        output,
                        "rollout_environments",
                        num_envs=config.num_envs,
                        **simulator.call("request", "check_reset").wait()[0],
                    )
            else:
                manifest = json.loads((Path(resume) / "manifest.json").read_text())
                assert manifest["initial_checkpoint"] == config.checkpoint
                checkpoints["sft"] = manifest["sft_checkpoint"]
                assert (Path(checkpoints["sft"]) / "model.safetensors").is_file()
                restored = learner.call("resume", resume).wait()[0]
                transitions = restored["transitions"]
                _event(output, "resume", **restored)
            rl_deadline = min(
                time.time() + config.rl_hours * 3600,
                deadline - (config.total_hours - config.prepare_hours - config.rl_hours) * 3600,
            )
            # A resumed learner must not repeat the initial training episodes.
            observations = simulator.call(
                "request",
                "reset",
                seeds=list(range(100_000 + transitions, 100_000 + transitions + config.num_envs)),
            ).wait()[0]
            buffered = 0
            while (
                transitions + config.num_envs <= config.max_transitions
                and time.time() < rl_deadline
            ):
                actions = learner.call("sample", observations).wait()[0]
                execute = min(
                    config.execute_steps, (config.max_transitions - transitions) // config.num_envs
                )
                result = simulator.call(
                    "request", "step", actions=actions[:, :execute], gamma=config.gamma
                ).wait()[0]
                counted = learner.call("record", result).wait()[0]
                transitions, buffered = counted["transitions"], counted["rollout_samples"]
                observations = result["observations"]
                done = np.flatnonzero(result["terminated"] | result["truncated"]).tolist()
                for i in done:
                    _event(output, "source_episode", **result["info"][i])
                if done:
                    observations = simulator.call("request", "reset", ids=done).wait()[0]
                if buffered >= config.global_rollout_batch_size:
                    updated = learner.call("update").wait()[0]
                    _event(output, "ppo", **updated)
                    buffered = 0
                    if updated["updates"] % config.checkpoint_every_updates == 0:
                        path = output / "checkpoints" / f"rl-step-{transitions:09d}"
                        _event(
                            output,
                            "checkpoint_periodic",
                            **learner.call("save", str(path)).wait()[0],
                        )
            discarded = learner.call("discard_rollout").wait()[0]
            _event(output, "rollout_budget_end", **discarded)
            assert discarded["updates"] > 0, "RL budget exhausted without a full PPO rollout"
            _event(output, "checkpoint_rl", **learner.call("save", checkpoints["rl"]).wait()[0])
            source_rows = []
            source_deadline = min(time.time() + 3600, deadline)
            for label in LABELS:
                learner.call("load", checkpoints[label], "paint").wait()
                rows = source_evaluation(simulator, learner, config, label, source_deadline)
                source_rows.extend(rows)
                _event(output, "source_evaluation", checkpoint=label, episodes=len(rows))
            (output / "source-eval/episodes.json").write_text(
                json.dumps(source_rows, indent=2) + "\n"
            )
            simulator.call("restart", config.libero_script, {}, config.libero_python).wait()
            summary = target_evaluation(simulator, learner, config, checkpoints, deadline)
            summary.update(
                elapsed_hours=(time.time() - started) / 3600,
                checkpoints=checkpoints,
                transitions=transitions,
                requested_paired_cases=40 * config.target_eval_rounds,
                complete=summary["paired_cases"] == 40 * config.target_eval_rounds,
            )
            (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
            _event(output, "finished", **summary)
            return summary
    except BaseException as error:
        _event(output, "failed", error_type=type(error).__name__, error=str(error))
        raise
