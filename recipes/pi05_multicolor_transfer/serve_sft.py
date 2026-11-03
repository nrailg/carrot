import argparse
import os
from dataclasses import replace

import numpy as np

from carrot.models.pi05.inference.websocket_policy_server import WebsocketPolicyServer
from carrot.trainer.rl.config import load_config
from carrot.trainer.rl.learner import LearnerWorker


class DiagnosticPolicy:
    def __init__(self, config: str, checkpoint: str) -> None:
        os.environ.update(LOCAL_RANK="0", MASTER_ADDR="127.0.0.1")
        self.learner = LearnerWorker(replace(load_config(config), learner_node_ip="127.0.0.1"))
        self.learner.setup()
        self.learner.resume(checkpoint)

    def infer(self, observation: dict) -> dict[str, np.ndarray]:
        return {"actions": self.learner.infer([observation])[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    policy = DiagnosticPolicy(args.config, args.checkpoint)
    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={"checkpoint": args.checkpoint, "weights": "fp32_learner", "ode_steps": 10},
    ).serve_forever()
