import argparse
import json
from pathlib import Path

import numpy as np
from eval_common import file_sha

from carrot.models.pi05.inference import create_libero_policy
from carrot.models.pi05.inference.websocket_policy_server import WebsocketPolicyServer


class PairedPolicy:
    def __init__(self, policy) -> None:
        self.policy = policy

    def infer(self, observation: dict) -> dict:
        observation = dict(observation)
        noise = np.asarray(observation.pop('eval_noise'), dtype=np.float32)
        return self.policy.infer(observation, noise=noise)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--norm-stats', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / 'config.json').read_text())
    metadata = {'checkpoint': str(args.checkpoint),
                'model_sha256': file_sha(args.checkpoint / 'model.safetensors'),
                'norm_stats_sha256': file_sha(Path(args.norm_stats)),
                'action_dim': config['action_dim'], 'action_horizon': 10,
                'ode_steps': 10, 'execute_steps': 5, 'precision': config['precision']}
    policy = create_libero_policy(args.checkpoint, device='cuda:0',
                                  tokenizer_path=args.tokenizer,
                                  norm_stats_path=args.norm_stats, num_steps=10)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'loaded': metadata}), flush=True)
    WebsocketPolicyServer(PairedPolicy(policy), host=args.host,
                          port=args.port, metadata=metadata).serve_forever()
