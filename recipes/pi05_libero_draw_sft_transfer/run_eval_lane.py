import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--norm-stats', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-gpu', type=int, required=True)
    parser.add_argument('--render-gpu', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    assert args.policy_gpu != args.render_gpu
    args.output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    server = None
    code = 1
    try:
        with (args.output / 'server.log').open('w') as log:
            server = subprocess.Popen([sys.executable, '-u', str(scripts / 'serve_eval.py'),
                '--checkpoint', str(args.checkpoint), '--tokenizer', args.tokenizer,
                '--norm-stats', args.norm_stats, '--host', '127.0.0.1', '--port', str(args.port),
                '--record', str(args.output / 'loaded.json')], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.policy_gpu)})
            deadline = time.monotonic() + 600
            while True:
                assert server.poll() is None, 'policy server exited; inspect server.log'
                assert time.monotonic() < deadline, 'policy server readiness timeout'
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/healthz', timeout=2) as response:
                        assert response.read() == b'OK\n'
                    break
                except urllib.error.URLError:
                    time.sleep(2)
            with (args.output / 'client.log').open('w') as client_log:
                result = subprocess.run(['/opt/venvs/openpi-libero/bin/python', '-u',
                    str(scripts / 'eval_libero.py'), '--port', str(args.port),
                    '--output', str(args.output / 'episodes')], stdout=client_log,
                    stderr=subprocess.STDOUT, check=False, env={**os.environ,
                        'CUDA_VISIBLE_DEVICES': str(args.render_gpu),
                        'MUJOCO_EGL_DEVICE_ID': str(args.render_gpu), 'MUJOCO_GL': 'egl',
                        'PYOPENGL_PLATFORM': 'egl', 'LIBERO_CONFIG_PATH': '/opt/libero',
                        'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1'})
            assert result.returncode == 0, f'evaluator exited {result.returncode}; inspect client.log'
            code = 0
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait(timeout=30)
        (args.output / 'status.json').write_text(json.dumps({'returncode': code,
            'complete': code == 0, 'finished_at': time.time()}) + '\n')


if __name__ == '__main__':
    main()
