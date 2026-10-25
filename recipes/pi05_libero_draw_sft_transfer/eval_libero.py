import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from eval_common import file_sha, paired_noise
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from robosuite.utils.transform_utils import quat2axisangle

LIMITS = {'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300, 'libero_10': 520}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()
    assert 1 <= args.trials <= 50
    args.output.mkdir(parents=True, exist_ok=False)
    client = WebsocketClientPolicy('127.0.0.1', args.port)
    metadata = client.get_server_metadata()
    (args.output / 'policy.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows = []
    for suite_name, limit in LIMITS.items():
        suite = benchmark.get_benchmark_dict()[suite_name]()
        assert suite.n_tasks == 10
        for task_id in range(10):
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            try:
                for episode in range(args.trials):
                    env.seed(7)
                    env.reset()
                    initial = np.asarray(initial_states[episode])
                    obs = env.set_init_state(initial_states[episode])
                    for _ in range(10):
                        obs, _, _, _ = env.step([0.] * 6 + [-1.])
                    plan = deque()
                    key = f'{suite_name}/{task_id}/{episode}'
                    noise_hash = hashlib.sha256()
                    writer = imageio.get_writer(args.output / f'{suite_name}-{task_id}.mp4', fps=10) if episode == 0 else None
                    chunk = 0
                    success = False
                    try:
                        for control in range(limit):
                            images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
                                np.ascontiguousarray(obs[name][::-1, ::-1]), 224, 224))
                                for name in ['agentview_image', 'robot0_eye_in_hand_image']]
                            if writer is not None:
                                writer.append_data(images[0])
                            if not plan:
                                noise = paired_noise(key, chunk, metadata['action_dim'])
                                noise_hash.update(noise.tobytes())
                                element = {'observation/image': images[0],
                                           'observation/wrist_image': images[1],
                                           'observation/state': np.concatenate((obs['robot0_eef_pos'],
                                                quat2axisangle(obs['robot0_eef_quat'].copy()),
                                                obs['robot0_gripper_qpos'])),
                                           'prompt': str(task.language), 'eval_noise': noise}
                                actions = np.asarray(client.infer(element)['actions'])
                                assert actions.shape == (10, 7) and np.isfinite(actions).all()
                                plan.extend(actions[:5])
                                chunk += 1
                            obs, _, done, _ = env.step(plan.popleft().tolist())
                            if done:
                                success = True
                                break
                    finally:
                        if writer is not None:
                            writer.close()
                    row = {'suite': suite_name, 'task_id': task_id, 'episode_idx': episode,
                           'success': success, 'controls': control + 1, 'calls': chunk,
                           'init_sha256': hashlib.sha256(initial.tobytes()).hexdigest(),
                           'bddl_sha256': file_sha(bddl), 'noise_key': key,
                           'noise_sha256': noise_hash.hexdigest(), 'env_seed': 7}
                    rows.append(row)
                    with (args.output / 'episodes.jsonl').open('a') as stream:
                        stream.write(json.dumps(row) + '\n')
                    print(json.dumps(row), flush=True)
            finally:
                env.close()
    assert len(rows) == 40 * args.trials
    (args.output / 'result.json').write_text(json.dumps({'complete': True, 'episodes': len(rows),
                                                       'successes': sum(r['success'] for r in rows)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
