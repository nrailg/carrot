import os
from pathlib import Path

import numpy as np
import torch

from carrot.models.pi05.inference import create_so101_policy
from examples.so101_real.config import load_config
from examples.so101_real.dataset import load_dataset_source


def test_so101_checkpoint_on_recorded_observation() -> None:
    # 用真实 SFT checkpoint 和原始数据帧验证独立重载、统计量与六维动作解码。
    checkpoint = os.environ["CARROT_SO101_CHECKPOINT"]
    dataset_root = os.environ["CARROT_SO101_DATASET"]
    assert torch.cuda.is_available(), "SO101 checkpoint test requires CUDA"
    policy = create_so101_policy(checkpoint, device="cuda:0", joint_units="normalized")
    horizon = policy.metadata["action_horizon"]
    assert horizon == 50 and policy.metadata["action_dim"] == 6
    config = load_config(Path("examples/so101_real/deployment.yaml"),
                         {"dataset_root": dataset_root,
                          "dataset_repo": "felixmayor/orange_cube_merged"})
    source = load_dataset_source(config, horizon)
    frame = source.read()
    assert frame is not None

    # 固定噪声使两次完整推理可比较，输入不包含示教 action。
    noise = np.random.default_rng(7).standard_normal((horizon, 32)).astype(np.float32)
    first = policy.infer(frame.request, noise=noise)
    second = policy.infer(frame.request, noise=noise)

    # 完整输出必须有限且可重复；该测试只证明推理契约，不代表任务成功率。
    assert first["actions"].dtype == np.float32
    assert first["actions"].shape == (50, 6)
    assert np.isfinite(first["actions"]).all()
    np.testing.assert_array_equal(first["actions"], second["actions"])
