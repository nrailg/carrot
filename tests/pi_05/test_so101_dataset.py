import os

from examples.so101_real.config import DeploymentConfig
from examples.so101_real.dataset import load_dataset_source


def test_real_so101_episode_observations() -> None:
    # 使用真实本地视频和 parquet，验证 episode 过滤与实际 LeRobot padding 契约。
    config = DeploymentConfig(dataset_root=os.environ["CARROT_SO101_DATASET"])
    source = load_dataset_source(config, 50)
    first = source.read()

    # 检查双相机解码及模型请求字段，不能把示教动作作为模型输入。
    assert first is not None and first.frame == 0
    assert first.request["observation/state"].shape == (6,)
    assert first.request["observation/image"].shape == (540, 960, 3)
    assert first.request["observation/wrist_image"].shape == (640, 480, 3)
    assert first.request["prompt"]
    assert "actions" not in first.request

    # 跳到末尾两帧，确认有效动作数逐步变为 2、1、0，没有跨 episode 读取。
    source.frame = len(source.dataset) - 2
    assert source.read().valid_steps == 2
    source.advance(1)
    assert source.read().valid_steps == 1
    source.advance(1)
    assert source.read() is None
