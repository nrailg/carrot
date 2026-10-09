from dataclasses import dataclass
from typing import Any, override

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotObservation
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep

assert (
    ProcessorStepRegistry.get("pi05_prepare_state_tokenizer_processor_step")
    is Pi05PrepareStateTokenizerProcessorStep
)


@dataclass
@ProcessorStepRegistry.register(name="arm_dance_state_jitter")
class StateJitterStep(ObservationProcessorStep):
    lower: list[float]
    upper: list[float]
    enabled: bool = True
    degrees: float = 3.0

    def __post_init__(self) -> None:
        limits = torch.tensor([self.lower, self.upper])
        assert limits.shape == (2, 5) and torch.isfinite(limits).all()
        assert (limits[0] < limits[1]).all() and self.degrees == 3.0

    @override
    def observation(self, observation: RobotObservation) -> RobotObservation:
        state = observation["observation.state"]
        assert state.shape[-1] == 6 and torch.isfinite(state).all()
        if not self.enabled:
            return observation
        state = state.clone()
        lower, upper = state.new_tensor(self.lower), state.new_tensor(self.upper)
        noise = torch.empty_like(state[..., :5]).uniform_(-self.degrees, self.degrees)
        state[..., :5] = torch.clamp(state[..., :5] + noise, min=lower, max=upper)
        return observation | {"observation.state": state}

    @override
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    @override
    def get_config(self) -> dict[str, Any]:
        return {"lower": self.lower, "upper": self.upper, "enabled": self.enabled, "degrees": self.degrees}
