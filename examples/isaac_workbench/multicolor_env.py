import json
import time
from pathlib import Path

import numpy as np
from isaacsim.core.api import World
from multicolor_logic import generate_task
from multicolor_scene import MulticolorWorkbench, PaintExpert
from PIL import Image
from scene import DT

from carrot.trainer.rl.data import reuse_demonstrations


class MulticolorEnv:
    """Synchronous vector environment; reset is explicit and never advances physics.

    Parameters
    ----------
    robot_usd : str
    num_envs : int
    max_episode_steps : int
    seed : int
        Training uses seeds below one million; evaluation supplies disjoint seeds explicitly.
    num_colors : int
    """

    def __init__(
        self,
        robot_usd: str,
        num_envs: int = 4,
        max_episode_steps: int = 4000,
        seed: int = 0,
        num_colors: int = 3,
    ) -> None:
        assert num_envs >= 1 and max_episode_steps > 0
        self.world = World(physics_dt=DT, rendering_dt=1 / 20, stage_units_in_meters=1)
        self.next_seed, self.num_colors = seed, num_colors
        self.max_episode_steps = max_episode_steps
        self.envs = [
            MulticolorWorkbench(Path(robot_usd), generate_task(seed + i, num_colors), self.world, i)
            for i in range(num_envs)
        ]
        self.world.reset()
        for env in self.envs:
            env.initialize()
        for _ in range(30):
            self.world.step(render=True)
        self.needs_reset = np.zeros(num_envs, dtype=bool)
        self.experts = [PaintExpert(env) for env in self.envs]
        self.next_seed += num_envs
        self.last_info: list[dict] = [{} for _ in self.envs]

    def observe(self) -> list[dict]:
        return [env.actor_observation() for env in self.envs]

    def reset(
        self,
        ids: list[int] | None = None,
        seeds: list[int] | None = None,
        num_colors: int | None = None,
    ) -> list[dict]:
        """Reset selected slots without advancing physics in any slot.

        Parameters
        ----------
        ids : list[int] | None
            None resets every slot.
        seeds : list[int] | None
            Explicit seeds may select evaluation tasks; automatic seeds stay below 1,000,000.
        num_colors : int | None
            Demonstration curriculum override; None uses the configured formal task.

        Returns
        -------
        list[dict]
            Observations for all slots, including those left unchanged.
        """
        ids = list(range(len(self.envs))) if ids is None else ids
        assert len(set(ids)) == len(ids) and all(0 <= i < len(self.envs) for i in ids)
        if seeds is None:
            seeds = list(range(self.next_seed, self.next_seed + len(ids)))
            self.next_seed += len(ids)
            assert self.next_seed < 1_000_000, "training seeds overlap evaluation partition"
        assert len(ids) == len(seeds)
        colors = self.num_colors if num_colors is None else num_colors
        for i, seed in zip(ids, seeds, strict=True):
            self.envs[i].reset_task(generate_task(seed, colors))
            self.experts[i] = PaintExpert(self.envs[i])
            self.needs_reset[i] = False
            self.last_info[i] = {}
        self.world.render()
        return self.observe()

    def step(self, actions: np.ndarray, gamma: float = 0.99) -> dict:
        """Execute at most one chunk, stopping all slots at the first subtask or episode boundary.

        Parameters
        ----------
        actions : numpy.ndarray
            Shape (env, requested_steps, 7), with requested_steps between one and five.
        gamma : float
            Per-control discount used for the returned within-chunk reward sum.

        Returns
        -------
        dict
            Pre-reset observations, actual durations, discounted rewards and explicit done masks.
        """
        assert actions.shape[0] == len(self.envs) and actions.shape[2] == 7
        assert 1 <= actions.shape[1] <= 5 and np.isfinite(actions).all()
        assert not self.needs_reset.any(), "reset terminal slots before stepping"
        rewards = np.zeros(len(self.envs), dtype=np.float32)
        for step in range(actions.shape[1]):
            for env, action in zip(self.envs, actions[:, step], strict=True):
                env.command(action)
            for tick in range(12):
                for env in self.envs:
                    env.servo()
                self.world.step(render=tick == 11)
                for env in self.envs:
                    env.read_contacts()
            results = [env.finish_control() for env in self.envs]
            reward, terminated, changed, self.last_info = map(list, zip(*results, strict=True))
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.array([env.controls >= self.max_episode_steps for env in self.envs])
            truncated &= ~terminated
            self.needs_reset = terminated | truncated
            rewards += gamma**step * np.asarray(reward)
            if any(changed) or self.needs_reset.any():
                break
        for env in self.envs:
            env.update_canvas()
        self.world.render()
        return {
            "observations": self.observe(),
            "rewards": rewards,
            "lengths": np.full(len(self.envs), step + 1, dtype=np.int64),
            "terminated": terminated,
            "truncated": truncated,
            "info": self.last_info,
        }

    def expert_actions(self) -> np.ndarray:
        return np.stack([expert.action() for expert in self.experts])[:, None]

    def snapshot(self, output: str) -> None:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        for i, env in enumerate(self.envs):
            Image.fromarray(env.rgb()).save(directory / f"env-{i}-base.png")
            Image.fromarray(env.actor_observation()["observation/wrist_image"]).save(
                directory / f"env-{i}-wrist.png"
            )
            env.canvas.image.save(directory / f"env-{i}-canvas.png")
        (directory / "metrics.json").write_text(json.dumps(self.last_info, indent=2) + "\n")

    def check_reset(self) -> dict:
        assert len(self.envs) >= 2, "partial-reset check needs at least two slots"
        before = [
            (
                e.robot.get_joint_positions().copy(),
                e.pens.get_world_poses()[0].copy(),
                e.tips.get_world_poses()[0].copy(),
                e.canvas.image.tobytes(),
                e.controls,
            )
            for e in self.envs[1:]
        ]
        self.reset([0])
        for env, saved in zip(self.envs[1:], before, strict=True):
            assert np.array_equal(env.robot.get_joint_positions(), saved[0])
            assert np.array_equal(env.pens.get_world_poses()[0], saved[1])
            assert np.array_equal(env.tips.get_world_poses()[0], saved[2])
            assert env.canvas.image.tobytes() == saved[3] and env.controls == saved[4]
        assert self.envs[0].canvas.contact_samples == 0 and self.envs[0].controls == 0
        return {"partial_reset": "PASS", "unchanged_slots": len(before)}

    def collect(
        self,
        output: str,
        successes: int,
        attempts: int,
        deadline: float,
        source: str | None = None,
    ) -> dict:
        """Record pre-action observations and retain only complete successful expert episodes.

        Parameters
        ----------
        output : str
        successes, attempts : int
            Required successful episodes and hard cap on started attempts, respectively.
        deadline : float
            Absolute Unix deadline; partial episodes are discarded and recorded as interrupted.
        source : str | None
            Explicit previous collection to reuse; only committed successes are copied.

        Returns
        -------
        dict
            Success/failure accounting; caller must reject an incomplete demonstration curriculum.
        """
        assert attempts >= successes >= len(self.envs)
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=False)
        buffers: list[list[dict]] = [[] for _ in self.envs]
        finished = saved = 0
        records = []
        if source is not None:
            reused = reuse_demonstrations(source, directory, successes)
            saved = finished = reused["saved"]
            records = reused["episodes"]
            self.next_seed = max(self.next_seed, reused["next_seed"])
            (directory / "reuse.json").write_text(json.dumps(reused, indent=2) + "\n")
        colors = min(3, 1 + saved * 3 // successes)
        observations = self.reset(num_colors=colors)
        started = saved + (len(self.envs) if saved < successes else 0)
        assert started <= attempts
        (directory / "collection.json").write_text(
            json.dumps(
                {
                    "saved": saved,
                    "completed": finished,
                    "started": started,
                    "required": successes,
                    "episodes": records,
                },
                indent=2,
            )
            + "\n"
        )
        while time.time() < deadline and saved < successes:
            actions = self.expert_actions()
            for i, obs in enumerate(observations):
                buffers[i].append(
                    {**obs, "actions": actions[i, 0].copy(), "phase": self.envs[i].schedule.key}
                )
            result = self.step(actions)
            observations = result["observations"]
            done = np.flatnonzero(result["terminated"] | result["truncated"]).tolist()
            for i in done:
                info = result["info"][i]
                records.append(info)
                finished += 1
                if info["success"] and saved < successes:
                    episode = directory / f"episode-{saved:04d}"
                    episode.mkdir()
                    for key in buffers[i][0]:
                        name = key.replace("observation/", "")
                        np.save(episode / (name + ".npy"), np.asarray([x[key] for x in buffers[i]]))
                    (episode / "result.json").write_text(json.dumps(info, indent=2) + "\n")
                    saved += 1
                buffers[i].clear()
            report = {
                "saved": saved,
                "completed": finished,
                "started": started,
                "required": successes,
                "episodes": records,
            }
            if done:
                (directory / "collection.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True)
                if started + len(done) > attempts or saved >= successes:
                    break
                colors = min(3, 1 + saved * 3 // successes)
                observations = self.reset(done, num_colors=colors)
                started += len(done)
        report = {
            "saved": saved,
            "completed": finished,
            "started": started,
            "required": successes,
            "interrupted": started - finished,
            "episodes": records,
        }
        (directory / "collection.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    def close(self) -> None:
        self.world.stop()
