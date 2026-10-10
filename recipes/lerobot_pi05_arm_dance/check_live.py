import pickle
import unittest
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from lerobot.async_inference.helpers import TimedAction
from lerobot.transport import services_pb2
from live import KEYS, predict


class Stub:
    def __init__(self, actions: list[TimedAction] | None) -> None:
        self.actions = actions
        self.observation = None

    def SendObservations(self, chunks: Iterable[Any], timeout: float) -> None:
        assert timeout == 10
        self.observation = pickle.loads(b"".join(chunk.data for chunk in chunks))

    def GetActions(self, request: Any, timeout: float) -> Any:
        assert timeout == 30
        data = b"" if self.actions is None else pickle.dumps(self.actions)
        return services_pb2.Actions(data=data)


class LiveChecks(unittest.TestCase):
    def test_transport_contract(self) -> None:
        # 验证 state、任务、零图像和步号在官方序列化前后不变，避免关节错序。
        state = dict(zip(KEYS, range(6), strict=True))
        stub = Stub([TimedAction(0.0, 17, torch.arange(6, dtype=torch.float32))])

        # 调用真实请求构造路径；替身仅替代网络，不连接机械臂。
        action, latency = predict(stub, state, 17)
        observation = stub.observation

        # 检查输入无 jitter、无视觉，以及输出保留关节顺序。
        assert observation.timestep == 17 and observation.must_go
        assert all(observation.observation[key] == value for key, value in state.items())
        assert observation.observation["task"] == "Dance with the arm"
        assert not np.any(observation.observation["wrist"])
        np.testing.assert_array_equal(action, np.arange(6))
        assert latency >= 0

    def test_invalid_response_fails(self) -> None:
        # 空响应、过期步号、错误维度或 NaN 必须在任何硬件动作前失败。
        state = dict.fromkeys(KEYS, 0.0)
        responses = [None, [], [TimedAction(0.0, 16, torch.zeros(6))],
                     [TimedAction(0.0, 17, torch.zeros(5))],
                     [TimedAction(0.0, 17, torch.full((6,), float("nan")))]]

        # 分别注入每种非法响应，要求请求路径显式断言失败。
        for response in responses:
            with self.subTest(response=response), self.assertRaises(AssertionError):
                predict(Stub(response), state, 17)


if __name__ == "__main__":
    unittest.main()
