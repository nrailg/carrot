import os
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TransferConfig:
    checkpoint: str
    tokenizer: str
    norm_stats: str
    robot_usd: str
    output: str
    simulator_output: str
    simulator_source: str
    simulator_node_ip: str
    learner_node_ip: str
    simulator_script: str
    libero_script: str
    libero_python: str = "/opt/venvs/openpi-libero/bin/python"
    num_nodes: int = 2
    simulator_bundle: int = 0
    learner_bundle: int = 1
    seed: int = 7
    num_envs: int = 4
    max_episode_steps: int = 4000
    num_colors: int = 3
    execute_steps: int = 5
    denoise_steps: int = 5
    noise_level: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    policy_lr: float = 5e-6
    value_lr: float = 1e-4
    sft_lr: float = 1e-5
    sft_steps: int = 500
    sft_batch_size: int = 16
    micro_batch_size: int = 2
    minibatch_size: int = 8
    rollout_chunks: int = 64
    checkpoint_every_updates: int = 10
    max_transitions: int = 200_000
    demo_successes: int = 64
    demo_attempts: int = 128
    source_eval_episodes: int = 8
    target_eval_rounds: int = 5
    prepare_hours: float = 6
    rl_hours: float = 10
    total_hours: float = 24

    def __post_init__(self) -> None:
        assert self.num_nodes in (1, 2)
        assert {self.simulator_bundle, self.learner_bundle} == {0, 1}
        assert (self.simulator_node_ip == self.learner_node_ip) == (self.num_nodes == 1)
        assert self.num_envs >= 2 and self.num_colors in (2, 3)
        assert 1 <= self.execute_steps <= 5 and self.denoise_steps >= 2
        assert self.noise_level > 0 and 0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1
        assert self.minibatch_size >= self.micro_batch_size >= 1
        assert self.minibatch_size % self.micro_batch_size == 0
        assert self.sft_batch_size % self.micro_batch_size == 0
        assert self.rollout_chunks * self.num_envs % self.minibatch_size == 0
        assert self.demo_attempts >= self.demo_successes >= self.num_envs
        assert self.source_eval_episodes % self.num_envs == 0
        assert 0 < self.prepare_hours + self.rl_hours < self.total_hours
        assert self.sft_steps > 0 and self.max_transitions > 0 and self.target_eval_rounds > 0
        assert self.checkpoint_every_updates > 0

    def validate_paths(self) -> None:
        for path in (
            Path(self.checkpoint) / "model.safetensors",
            Path(self.tokenizer),
            Path(self.norm_stats),
        ):
            assert path.exists(), f"missing local resource: {path}"


def load_config(path: str) -> TransferConfig:
    with Path(path).open() as stream:
        data = yaml.safe_load(os.path.expandvars(stream.read()))
    assert isinstance(data, dict)
    assert "$" not in str(data), "configuration contains unresolved environment variables"
    config = TransferConfig(**data)
    config.validate_paths()
    return config


def save_config(config: TransferConfig, output: Path) -> None:
    output.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
