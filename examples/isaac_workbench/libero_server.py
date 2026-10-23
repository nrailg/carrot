import argparse
import hashlib
import json
import os
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch
from openpi_client import image_tools

parser = argparse.ArgumentParser()
parser.add_argument("--socket", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--gpu", type=int, required=True)
args = parser.parse_args()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

LIMITS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def observation(raw: dict, prompt: str) -> dict:
    quat = np.asarray(raw["robot0_eef_quat"]).copy()
    quat[3] = np.clip(quat[3], -1, 1)
    denominator = np.sqrt(1 - quat[3] ** 2)
    rotvec = (
        np.zeros(3)
        if np.isclose(denominator, 0)
        else quat[:3] * 2 * np.arccos(quat[3]) / denominator
    )
    images = [
        image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(raw[key][::-1, ::-1]), 224, 224)
        )
        for key in ("agentview_image", "robot0_eye_in_hand_image")
    ]
    return {
        "observation/image": images[0],
        "observation/wrist_image": images[1],
        "observation/state": np.concatenate(
            (raw["robot0_eef_pos"], rotvec, raw["robot0_gripper_qpos"])
        ).astype(np.float32),
        "prompt": prompt,
    }


class LiberoEnv:
    def __init__(self) -> None:
        self.env = None
        self.done = True

    def reset(self, suite: str, task_id: int, episode: int, seed: int) -> dict:
        assert suite in LIMITS and 0 <= task_id < 10 and episode >= 0
        if self.env is not None:
            self.env.close()
        np.random.seed(seed)
        tasks = benchmark.get_benchmark_dict()[suite]()
        assert tasks.n_tasks == 10
        task = tasks.get_task(task_id)
        state_path = (
            Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        )
        states = torch.load(state_path, map_location="cpu", weights_only=False)
        assert episode < len(states)
        state = np.asarray(states[episode])
        self.info = {
            "suite": suite,
            "task": task_id,
            "episode": episode,
            "seed": seed,
            "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        }
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(
                Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            ),
            camera_heights=256,
            camera_widths=256,
        )
        self.env.seed(seed)
        self.env.reset()
        self.raw = self.env.set_init_state(state)
        for _ in range(10):
            self.raw, _, _, _ = self.env.step([0.0] * 6 + [-1.0])
        self.prompt, self.limit = str(task.language), LIMITS[suite]
        self.steps, self.done = 0, False
        return {"observations": [observation(self.raw, self.prompt)], "info": self.info}

    def step(self, actions: np.ndarray) -> dict:
        assert self.env is not None and not self.done
        assert actions.ndim == 3 and actions.shape[0] == 1 and actions.shape[2] == 7
        assert 1 <= actions.shape[1] <= 5 and np.isfinite(actions).all()
        success = False
        for action in actions[0]:
            self.raw, _, success, _ = self.env.step(action.tolist())
            self.steps += 1
            self.done = bool(success or self.steps >= self.limit)
            if self.done:
                break
        return {
            "observations": [observation(self.raw, self.prompt)],
            "done": self.done,
            "info": {**self.info, "success": bool(success), "controls": self.steps},
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()


assert json.loads(args.config) == {}, "LIBERO evaluator accepts no training overrides"
environment = LiberoEnv()
listener = Listener(args.socket, family="AF_UNIX")
connection = listener.accept()
connection.send({"ready": True})
try:
    while True:
        method, kwargs = connection.recv()
        if method == "close":
            connection.send({"ok": True, "result": None})
            break
        methods = {"reset": environment.reset, "step": environment.step}
        assert method in methods
        try:
            result = methods[method](**kwargs)
        except BaseException:
            connection.send({"ok": False, "traceback": traceback.format_exc()})
            raise
        connection.send({"ok": True, "result": result})
finally:
    environment.close()
    connection.close()
    listener.close()
