from pathlib import Path

import numpy as np
from isaacsim.core.api import World
from PIL import Image
from scene import DT, Workbench
from tasks import CodingTask
from vm import VmPool


class VectorWorkbench:
    """Batched reset/step contract, with explicit partial reset and no automatic reset.

    Parameters
    ----------
    task : str
        computer or paint. All scenes share one Isaac physics/rendering process.
    references : list[PIL.Image.Image]
        One independent target per environment; returned as policy observations.
    coding_tasks : list[CodingTask] | None
        Required for computer. Test cases remain outside policy observations.
    max_episode_steps : int
        Truncation limit in control steps. Terminal observations are returned unchanged;
        reset(done_indices) is required before stepping again.
    """

    def __init__(
        self,
        task: str,
        robot_usd: Path,
        references: list[Image.Image],
        output: Path,
        guest_dir: Path | None = None,
        coding_tasks: list[CodingTask] | None = None,
        accelerator: str = "kvm",
        max_episode_steps: int = 4000,
        resolution: tuple[int, int] = (320, 240),
    ) -> None:
        assert task in ("computer", "paint") and references and max_episode_steps > 0
        self.task, self.num_envs = task, len(references)
        self.action_dim = 18 if task == "computer" else 9
        self.references = [image.convert("RGB").copy() for image in references]
        self.max_episode_steps = max_episode_steps
        self.vms = None
        self.world = None
        self.environments: list[Workbench] = []
        self.counts = np.zeros(self.num_envs, dtype=int)
        self.scores = np.zeros(self.num_envs)
        self.needs_reset = np.zeros(self.num_envs, dtype=bool)
        self.event_indices = np.zeros(self.num_envs, dtype=int)
        self.grades = [{} for _ in references]
        try:
            if task == "computer":
                assert guest_dir is not None and coding_tasks is not None
                assert len(coding_tasks) == self.num_envs
                self.vms = VmPool(guest_dir, output / "vms", coding_tasks, accelerator)
                displays = self.vms.screens()
            else:
                displays = self.references
            self.world = World(physics_dt=DT, rendering_dt=1 / 60, stage_units_in_meters=1.0)
            for i, display in enumerate(displays):
                self.environments.append(
                    Workbench(task, robot_usd, display, self.world, i, resolution)
                )
            self.world.reset()
            for env in self.environments:
                env.initialize()
            for _ in range(30):
                self.world.step(render=True)
            self.reference_pixels = (
                np.stack([np.asarray(im.resize(resolution)) for im in self.references])
                if task == "paint"
                else None
            )
        except BaseException:
            self.close()
            raise

    def observe(self) -> dict[str, np.ndarray]:
        observation = {
            "rgb": np.stack([env.rgb() for env in self.environments]),
            "joint_positions": np.stack(
                [env.robot.get_joint_positions().reshape(-1) for env in self.environments]
            ),
            "joint_velocities": np.stack(
                [env.robot.get_joint_velocities().reshape(-1) for env in self.environments]
            ),
        }
        if self.reference_pixels is not None:
            observation["reference_rgb"] = self.reference_pixels.copy()
        return observation

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        assert actions.shape == (self.num_envs, self.action_dim) and np.isfinite(actions).all()
        assert not self.needs_reset.any(), "call reset(done_indices) before another step"
        assert self.world is not None
        for env, action in zip(self.environments, actions, strict=True):
            env.apply(action)
        for tick in range(4):
            self.world.step(render=tick == 3)
            for env in self.environments:
                env.read_contacts()
        for env, action in zip(self.environments, actions, strict=True):
            env.observation(action)
        self.counts += 1
        if self.vms is not None:
            batches = [
                env.events[self.event_indices[i] :] for i, env in enumerate(self.environments)
            ]
            self.grades = self.vms.step(batches, self.environments[0].keys)
            self.event_indices = np.array([len(env.events) for env in self.environments])
            if any(batches):
                for env, screen in zip(self.environments, self.vms.screens(), strict=True):
                    env.monitor.update(screen)
            scores = np.array([grade["passed"] / grade["total"] for grade in self.grades])
            terminated = np.array([grade["success"] for grade in self.grades])
            terminated &= np.array([not env.switches.down.any() for env in self.environments])
        else:
            scores = np.array(
                [
                    env.canvas.score(ref)
                    for env, ref in zip(self.environments, self.references, strict=True)
                ]
            )
            for env in self.environments:
                env.update_canvas()
            # Success is only final after lifting the brush, not halfway through a stroke.
            terminated = (scores >= 0.72) & np.array(
                [
                    env.tool_pose()[0][2] > 0.175 and env.last_force < 0.02
                    for env in self.environments
                ]
            )
        reward = scores - self.scores
        self.scores = scores
        truncated = (self.counts >= self.max_episode_steps) & ~terminated
        self.needs_reset = terminated | truncated
        # Updated monitor/canvas textures must be rendered before camera observations are exposed.
        self.world.render()
        return (
            self.observe(),
            reward,
            terminated,
            truncated,
            {"scores": scores.copy(), "grades": self.grades.copy()},
        )

    def reset(self, env_ids: list[int] | None = None) -> tuple[dict, dict]:
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        assert len(set(ids)) == len(ids) and all(0 <= i < self.num_envs for i in ids)
        if self.vms is not None:
            self.vms.reset(ids)
        for i in ids:
            self.environments[i].reset_state()
            if self.vms is not None:
                self.environments[i].monitor.update(self.vms.guests[i].screenshot())
            self.counts[i] = 0
            self.scores[i] = 0
            self.needs_reset[i] = False
            self.event_indices[i] = 0
            self.grades[i] = {}
        # Render only: resetting env i must not advance another environment's physics or VM.
        self.world.render()
        return self.observe(), {"reset_env_ids": ids}

    def close(self) -> None:
        if self.vms is not None:
            self.vms.close()
        if self.world is not None:
            self.world.stop()
