import os

import numpy as np
import pytest
import torch

from carrot.models.pi05.inference import create_trained_policy


def test_real_checkpoint_inference() -> None:
    # 使用真实 SFT export 跑完整采样和动作解码，验证固定 noise 下输出可重复。
    checkpoint = os.environ.get("CARROT_PI05_INFERENCE_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("set CARROT_PI05_INFERENCE_CHECKPOINT to a RoboTwin SFT export")
    if not torch.cuda.is_available():
        pytest.fail("checkpoint smoke requires CUDA")

    # checkpoint 必须携带训练时的 tokenizer 和 quantile stats；不回退到基础模型资产。
    policy = create_trained_policy(checkpoint, device="cuda:0")
    rng = np.random.default_rng(7)
    obs = {
        "state": np.zeros(14, dtype=np.float32),
        "images": {
            name: rng.integers(0, 256, (3, 224, 224), dtype=np.uint8)
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
        "prompt": "pick up the cup",
    }
    horizon = policy.metadata["action_horizon"]
    noise = rng.standard_normal((horizon, 32)).astype(np.float32)

    # 重复同一输入，两次都执行全部 denoising steps。
    first = policy.infer(obs, noise=noise)
    second = policy.infer(obs, noise=noise)

    # 这是推理 smoke，不能据此推断任务成功率；输出必须是可执行的 14 维 float32。
    assert first["actions"].shape == (horizon, 14)
    assert first["actions"].dtype == np.float32
    assert np.isfinite(first["actions"]).all()
    assert first["policy_timing"]["infer_ms"] > 0
    np.testing.assert_array_equal(first["actions"], second["actions"])
