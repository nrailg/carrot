import argparse
import hashlib
import json
import pickle
import shutil
import time
from concurrent import futures
from pathlib import Path
from typing import Any, override

import grpc
import torch
from configuration_pi05_control import PI05ControlConfig
from evaluate import verify_weights
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc
from lerobot.utils.random_utils import set_seed
from state_jitter import StateJitterStep


class ArmDanceServer(PolicyServer):
    def __init__(self, config: PolicyServerConfig, checkpoint: Path, output: Path) -> None:
        super().__init__(config)
        self.checkpoint = checkpoint
        self.output = output

    @override
    def SendPolicyInstructions(self, request: Any, context: Any) -> Any:
        specs = pickle.loads(request.data)
        assert isinstance(specs, RemotePolicyConfig)
        assert specs.policy_type == "pi05_control" and specs.device == "cuda"
        assert Path(specs.pretrained_name_or_path) == self.checkpoint
        assert specs.actions_per_chunk == 1 and not specs.rename_map
        result = super().SendPolicyInstructions(request, context)
        assert self.policy.config.chunk_size == 10 and self.policy.config.n_action_steps == 1
        assert self.policy.config.num_inference_steps == 10
        assert all(not step.enabled for step in self.preprocessor.steps
                   if isinstance(step, StateJitterStep))
        verified = verify_weights(self.policy.state_dict(), self.checkpoint)
        assert verified == 813
        set_seed(1000)
        (self.output / "loaded.json").write_text(json.dumps({
            "verified_tensors": verified,
            "checkpoint": str(self.checkpoint),
            "vision": False,
            "state_jitter": False,
        }, indent=2) + "\n")
        return result

    @override
    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        images, masks = self.policy._preprocess_images(observation)
        assert len(images) == 3 and all(torch.count_nonzero(image) == 0 for image in images)
        assert all(not mask.any() for mask in masks)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            action = super()._get_action_chunk(observation)
        assert action.shape == (1, 1, 6) and torch.isfinite(action).all()
        return action

    @override
    def _predict_action_chunk(self, observation: TimedObservation) -> list[TimedAction]:
        start = time.perf_counter()
        actions = super()._predict_action_chunk(observation)
        assert len(actions) == 1 and torch.isfinite(actions[0].action).all()
        row = {
            "step": observation.timestep,
            "state": {k: v for k, v in observation.observation.items() if k.endswith(".pos")},
            "action": actions[0].action.tolist(),
            "seconds": time.perf_counter() - start,
        }
        with (self.output / "predictions.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    assert PI05ControlConfig is not None
    args.output.mkdir(parents=True, exist_ok=False)
    exported = args.output / "inference_model"
    exported.mkdir()
    for source in args.checkpoint.iterdir():
        if source.is_file():
            if source.suffix == ".safetensors":
                (exported / source.name).symlink_to(source.resolve())
            else:
                shutil.copy2(source, exported / source.name)
    config_path = exported / "policy_preprocessor.json"
    config = json.loads(config_path.read_text())
    jitter = [step for step in config["steps"]
              if step["registry_name"] == "arm_dance_state_jitter"]
    assert len(jitter) == 1 and jitter[0]["config"]["enabled"] is True
    jitter[0]["config"]["enabled"] = False
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    sources = {}
    for source in Path(__file__).parent.glob("*.py"):
        sources[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (args.output / "source_sha256.json").write_text(json.dumps(sources, indent=2) + "\n")
    SUPPORTED_POLICIES.append("pi05_control")
    policy_server = ArmDanceServer(
        PolicyServerConfig(host="0.0.0.0", port=args.port, fps=15), exported, args.output,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    assert server.add_insecure_port(f"0.0.0.0:{args.port}") == args.port
    server.start()
    print(f"OFFICIAL POLICY SERVER READY {exported}", flush=True)
    try:
        server.wait_for_termination(timeout=1200)
    finally:
        server.stop(grace=5).wait()
        policy_server.stop()


if __name__ == "__main__":
    main()
