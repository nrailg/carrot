"""Independently verify stride-five feedback and executed-action metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from recipes.pi05_sft_so101_fit_validation.reevaluate import write_json

MODES = ("recorded_state", "predicted_action_state")


def check(values: list[np.ndarray], reported: dict) -> None:
    absolute = np.abs(np.concatenate(values))
    assert len(absolute) == reported["valid_action_rows"]
    for key, computed in (("joint_mae", absolute.mean(0)),
                          ("joint_p95", np.quantile(absolute, .95, axis=0)),
                          ("joint_max", absolute.max(0))):
        np.testing.assert_allclose(computed, reported[key], rtol=1e-6, atol=1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.source_data.rglob("*.parquet"))
    table = pa.concat_tables([pq.read_table(p, columns=["index", "episode_index",
                                                     "observation.state", "action"])
                              for p in paths])
    np.testing.assert_array_equal(np.asarray(table["index"]), np.arange(264))
    np.testing.assert_array_equal(np.asarray(table["episode_index"]), np.zeros(264))
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    metrics = json.loads((args.output / "metrics.json").read_text())
    assert metrics["vision"] is False
    assert metrics["chains"] == 8 and metrics["chunks_per_chain"] == 53
    assert metrics["horizon"] == metrics["nfe"] == 10 and metrics["execute_steps"] == 5
    assert len(list((args.output / "predictions").glob("*.npz"))) == 424
    assert {(r["chain"], r["chunk"], r["frame"], r["seed"]) for r in metrics["per_sample"]} == {
        (rep, depth, frame, 1000 + frame + rep * 100000)
        for rep in range(8) for depth, frame in enumerate(range(0, 264, 5))
    }
    collected = {mode: {depth: [] for depth in range(53)} for mode in MODES}
    endpoints = {mode: {depth: [] for depth in range(53)} for mode in MODES}
    inputs = {mode: {depth: [] for depth in range(53)} for mode in MODES}
    hashes = set()
    for rep in range(8):
        previous = None
        for depth, frame in enumerate(range(0, 264, 5)):
            path = args.output / "predictions" / f"chain_{rep:02d}_chunk_{depth:02d}.npz"
            with np.load(path) as f:
                saved = dict(f)
            reference = actions[np.minimum(np.arange(frame, frame + 10), 263)]
            valid = np.arange(10) < min(10, 264 - frame)
            executed = valid & (np.arange(10) < 5)
            np.testing.assert_array_equal(saved["reference"], reference)
            np.testing.assert_array_equal(saved["valid"], valid)
            np.testing.assert_array_equal(saved["executed_valid"], executed)
            np.testing.assert_array_equal(saved["recorded_state"], states[frame])
            seed = 1000 + frame + rep * 100000
            gaussian = np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
            noise = torch.from_numpy(gaussian).to(torch.bfloat16).float().numpy()
            np.testing.assert_array_equal(saved["noise"], noise)
            hashes.add(hashlib.sha256(noise.tobytes()).hexdigest())
            expected = {mode: states[frame].copy() for mode in MODES}
            if depth:
                expected["predicted_action_state"] = previous["predicted_predicted_action_state"][4]
            count = int(executed.sum())
            for mode in MODES:
                np.testing.assert_array_equal(saved[f"input_{mode}"], expected[mode])
                predicted = saved[f"predicted_{mode}"]
                assert predicted.shape == (10, 6) and np.isfinite(predicted).all()
                collected[mode][depth].append((predicted - reference)[executed])
                endpoints[mode][depth].append((predicted[count - 1] - reference[count - 1])[None])
                inputs[mode][depth].append((saved[f"input_{mode}"] - states[frame])[None])
                if not depth:
                    np.testing.assert_array_equal(predicted, saved["predicted_recorded_state"])
            previous = saved
    assert len(hashes) == 424
    for mode in MODES:
        result = metrics["modes"][mode]
        for depth in range(53):
            check(collected[mode][depth], result["chunk_errors"][str(depth)])
            check(endpoints[mode][depth], result["endpoint_errors"][str(depth)])
            check(inputs[mode][depth], result["input_state_errors"][str(depth)])
        values = [np.concatenate(v) for v in collected[mode].values()]
        assert len(np.concatenate(values)) == 264 * 8
        check(values, result["whole_episode"])
    write_json(args.output / "independent_validation.json", {
        "status": "PASS", "NPZ": 424, "unique_noise": len(hashes),
        "executed_actions_per_mode": 264 * 8,
        "checks": "raw reference/state/padding/noise; stride5 endpoint feedback at index4; "
                  "two matched chain heads; all MAE/P95/max",
        "raw_parquet_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "raw_states": states.tolist(), "raw_actions": actions.tolist(),
    })
    print("INDEPENDENT_CHUNK_ROLLOUT_PASS", "NPZ", 424, flush=True)


if __name__ == "__main__":
    main()
