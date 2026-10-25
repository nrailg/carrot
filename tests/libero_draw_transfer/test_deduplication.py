import json
import shutil

import pytest
from copy_collection import audit_episodes


def write_collection(root) -> None:
    for index in range(64):
        episode = root / "demos" / f"episode-{index:04d}"
        episode.mkdir(parents=True)
        (episode / "result.json").write_text(
            json.dumps(
                {
                    "seed": 200000 + index,
                    "success": True,
                    "picks": 3,
                    "returns": 3,
                    "drops": 0,
                }
            )
        )
        for name in ("actions", "state"):
            (episode / f"{name}.npy").write_bytes(f"{name}-{index}".encode())


def test_distinct_episodes_pass(tmp_path) -> None:
    # 验证64条独立示范能通过；此检查只读文件字节，数组结构由dataset契约另测。
    # 用不同字节构造完整序列指纹，避免把共同prompt或局部静止帧作为重复依据。
    write_collection(tmp_path)
    report = audit_episodes(tmp_path)
    # 报告必须保留全部episode身份，便于追溯发布的数据。
    assert report["passed"] and len(report["episodes"]) == 64
    assert report["unique_seeds"] == report["unique_trajectories"] == 64


@pytest.mark.parametrize("duplicate", ["seed", "actions", "state"])
def test_duplicate_episode_identity_is_rejected(tmp_path, duplicate) -> None:
    # 防止不同目录名或不同seed掩盖重复轨迹，也防止同seed重复采集混入数据。
    # 只复制一个身份维度，其余63条和其他维度仍保持唯一。
    write_collection(tmp_path)
    source, target = [tmp_path / "demos" / f"episode-{i:04d}" for i in (0, 1)]
    if duplicate == "seed":
        info = json.loads((target / "result.json").read_text())
        info["seed"] = 200000
        (target / "result.json").write_text(json.dumps(info))
    else:
        shutil.copyfile(source / f"{duplicate}.npy", target / f"{duplicate}.npy")
    # 发布前必须报错，不能悄悄少采一条后继续训练。
    with pytest.raises(AssertionError, match=f"duplicate {duplicate}"):
        audit_episodes(tmp_path)
