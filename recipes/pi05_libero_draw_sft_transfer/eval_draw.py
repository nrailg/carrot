import argparse
import hashlib
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from eval_common import file_sha, paired_noise
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from carrot.trainer.rl.simulator import SimulatorWorker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--robot-usd', required=True)
    parser.add_argument('--camera', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    (args.output / 'policy.json').write_text(json.dumps(metadata, indent=2) + '\n')
    camera = json.loads(args.camera.read_text())
    (args.output / 'camera.json').write_text(json.dumps(camera, indent=2) + '\n')
    source = Path(__file__).resolve().parents[2]
    os.environ.update(LOCAL_RANK='0', MASTER_ADDR='127.0.0.1')
    simulator = SimulatorWorker(str(source / 'examples/isaac_workbench/multicolor_server.py'),
        {'robot_usd': args.robot_usd, 'num_envs': 1, 'max_episode_steps': 4000,
         'num_colors': 3, 'seed': 1000000, 'teacher_mode': 'events', 'camera': camera},
        str(args.output / 'renderer'), '127.0.0.1', python='/opt/venvs/carrot/bin/python')
    simulator.setup()
    rows = []
    try:
        for seed in range(1000000, 1000032):
            observations = simulator.request('reset', seeds=[seed], num_colors=3)
            prompt = observations[0]['prompt']
            initial = hashlib.sha256(observations[0]['observation/state'].tobytes()).hexdigest()
            key = f'draw/{seed}'
            noise_hash = hashlib.sha256()
            writer = imageio.get_writer(args.output / f'{seed}.mp4', fps=4) if seed < 1000004 else None
            controls = 0
            try:
                for chunk in range(4000):
                    assert observations[0]['prompt'] == prompt
                    noise = paired_noise(key, chunk, metadata['action_dim'])
                    noise_hash.update(noise.tobytes())
                    actions = np.asarray(client.infer({**observations[0], 'eval_noise': noise})['actions'])
                    assert actions.shape == (10, 7) and np.isfinite(actions).all()
                    if writer is not None:
                        writer.append_data(observations[0]['observation/image'])
                    result = simulator.request('step', actions=actions[None, :5])
                    observations = result['observations']
                    controls += int(result['lengths'][0])
                    if chunk % 100 == 0:
                        print(json.dumps({'seed': seed, 'controls': controls, 'info': result['info'][0]}), flush=True)
                    if bool(result['terminated'][0] or result['truncated'][0]):
                        break
                else:
                    assert False, f'no terminal result for seed {seed}'
            finally:
                if writer is not None:
                    writer.close()
            row = {**result['info'][0], 'seed': seed, 'controls': controls, 'calls': chunk + 1,
                   'terminated': bool(result['terminated'][0]), 'truncated': bool(result['truncated'][0]),
                   'init_sha256': initial, 'camera_sha256': file_sha(args.camera),
                   'noise_key': key, 'noise_sha256': noise_hash.hexdigest(), 'prompt': prompt}
            rows.append(row)
            with (args.output / 'episodes.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps({'episode': row}), flush=True)
        assert len(rows) == 32 and len({r['seed'] for r in rows}) == 32
        (args.output / 'result.json').write_text(json.dumps({'complete': True, 'episodes': 32,
            'successes': sum(r['success'] for r in rows), 'mean_iou': float(np.mean([r['iou'] for r in rows]))}, indent=2) + '\n')
    finally:
        simulator.teardown()


if __name__ == '__main__':
    main()
