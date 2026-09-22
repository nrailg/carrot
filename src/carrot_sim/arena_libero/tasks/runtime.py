from dataclasses import asdict
from typing import Never, override

from isaaclab.envs import mdp as lab_mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.configclass import make_configclass

from carrot_sim.arena_libero.robot import robot_selection
from carrot_sim.arena_libero.task import RewardsCfg
from carrot_sim.arena_libero.tasks import mdp
from carrot_sim.arena_libero.tasks.spec import TaskSpec


class LiberoTask(TaskBase):
    def __init__(self, spec: TaskSpec, max_episode_steps: int) -> None:
        super().__init__(
            episode_length_s=max_episode_steps / 50,
            task_description=spec.language,
        )
        self.spec = spec

    @override
    def get_scene_cfg(self) -> None:
        return None

    @override
    def get_events_cfg(self) -> object:
        terms = [
            (
                "defaults",
                EventTermCfg,
                EventTermCfg(func=lab_mdp.reset_scene_to_default, mode="reset"),
            )
        ]
        for obj in self.spec.objects:
            if obj.xy_noise:
                terms.append(
                    (
                        obj.name,
                        EventTermCfg,
                        EventTermCfg(
                            func=lab_mdp.reset_root_state_uniform,
                            mode="reset",
                            params={
                                "asset_cfg": SceneEntityCfg(obj.name),
                                "pose_range": {
                                    "x": (-obj.xy_noise, obj.xy_noise),
                                    "y": (-obj.xy_noise, obj.xy_noise),
                                },
                                "velocity_range": {},
                            },
                        ),
                    )
                )
        return make_configclass("TaskResetCfg", terms)()

    @override
    def get_termination_cfg(self) -> object:
        goal = self.spec.goal
        target = SceneEntityCfg(goal.target)
        support = SceneEntityCfg(
            goal.support,
            body_names=[goal.support_body] if goal.support_body else None,
            joint_names=[goal.joint] if goal.joint else None,
        )
        terms = [
            (
                "success",
                TerminationTermCfg,
                TerminationTermCfg(
                    func=mdp.success,
                    params={
                        "goal": asdict(goal),
                        "target_cfg": target,
                        "support_cfg": support,
                        "robot_cfg": robot_selection(),
                    },
                ),
            ),
            (
                "dropped",
                TerminationTermCfg,
                TerminationTermCfg(func=mdp.dropped, params={"target_cfg": target}),
            ),
            (
                "time_out",
                TerminationTermCfg,
                TerminationTermCfg(func=lab_mdp.time_out, time_out=True),
            ),
        ]
        return make_configclass("TaskTerminationsCfg", terms)()

    @override
    def get_rewards_cfg(self) -> RewardsCfg:
        return RewardsCfg()

    @override
    def get_metrics(self) -> list[MetricBase]:
        return []

    @override
    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> Never:
        raise NotImplementedError("Mimic demonstrations are not provided for these RL tasks")
