import os
from pathlib import Path

import numpy as np
import yaml

from carrot.data.so101 import build_dataset
from examples.so101_real.config import load_config
from examples.so101_real.dataset import load_dataset_source


def test_real_knock_down_the_cylinder_source_values() -> None:
    # 同一真实录制帧经训练和client入口，必须得到一致的原始状态和reference。
    root = os.environ["CARROT_SO101_DATASET"]
    config = load_config(Path("examples/so101_real/knock_down_the_cylinder.yaml"),
                         {"dataset_root": root})
    training_config = yaml.safe_load(Path(
        "recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/train.yaml",
    ).read_text())["dataset"]["factory_kwargs"]
    spec = build_dataset(**(training_config | {"root": root}))
    assert len(spec.dataset) == 354

    # 首帧及episode0末帧包含不同padding长度，逐一核对原值、RGB和有效动作对齐。
    source = load_dataset_source(config, 50)
    for index in (0, len(source.dataset) - 1):
        source.frame = index
        frame = source.read()
        sample = spec.dataset[index]
        np.testing.assert_array_equal(
            frame.request["observation/state"], sample["observation/state"],
        )
        np.testing.assert_array_equal(frame.reference, sample["actions"][:frame.valid_steps])
        np.testing.assert_array_equal(
            frame.request["observation/wrist_image"],
            np.asarray(sample["observation/wrist_image"]).transpose(1, 2, 0),
        )
        raw = spec.dataset.source[index]
        np.testing.assert_allclose(sample["observation/state"][:5],
                                   np.asarray(raw["observation.state"])[:5], atol=1e-7)
        assert sample["observation/state"][5] == np.float32(raw["observation.state"][5])

    # 全部位置/尺度统计保留原值，count保持354，保证export与源数据一致。
    for feature, stats in (("observation.state", spec.state_stats), ("action", spec.action_stats)):
        raw_stats = spec.dataset.source.meta.stats[feature]
        for name, value in stats.items():
            if name == "count":
                np.testing.assert_array_equal(value, raw_stats[name])
            else:
                np.testing.assert_array_equal(
                    value, raw_stats[name],
                )
