import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from multicolor_controls import (
    EEF_OFFSET,
    EEF_ROTATION,
    LIBERO_BASE,
    command_target,
    libero_state,
    remove_sensitive_environment,
)
from multicolor_logic import (
    ColorCanvas,
    PaintSchedule,
    Stroke,
    color_iou,
    generate_task,
    render_strokes,
)
from PIL import Image
from scipy.spatial.transform import Rotation

from carrot.distributed.placement import PlacementSpec
from carrot.distributed.ray_runtime import RayRuntime
from carrot.trainer.rl.algorithms import (
    chunk_logprob,
    duration_gae,
    flow_transition,
    gaussian_logprob,
    ppo_loss,
)
from carrot.trainer.rl.data import Demonstrations
from carrot.trainer.rl.evaluation import LABELS, paired_summary


def test_simulator_environment_excludes_credentials() -> None:
    # 仿真只使用本地资产，Kit 子进程不得继承会被启动诊断输出的服务凭据。
    environment = {"PATH": "/fixture/bin", "CUDA_VISIBLE_DEVICES": "1", "TEST_TOKEN": "fixture"}
    with patch.dict(os.environ, environment, clear=True):
        remove_sensitive_environment()
        # 只去掉凭据类变量，设备映射和解释器路径必须保留。
        assert dict(os.environ) == {"PATH": "/fixture/bin", "CUDA_VISIBLE_DEVICES": "1"}


@pytest.mark.parametrize("index", range(5))
def test_flow_sde_scalar_oracle(index: int) -> None:
    # 用独立标量展开检查全部去噪位置，尤其 t=1 分母与最后一步噪声。
    latent = torch.tensor([[[0.7, -0.4]]])
    velocity = torch.tensor([[[0.3, 0.8]]])
    time, delta = 1 - index / 5, 1 / 5
    denominator = 0.8 if index == 0 else time
    sigma = 0.5 * math.sqrt(time / (1 - denominator))
    expected = []
    for x, v in zip(latent.flatten().tolist(), velocity.flatten().tolist(), strict=True):
        expected.append(
            (x - time * v) * (1 - time + delta)
            + (x + (1 - time) * v) * (time - delta - sigma**2 * delta / (2 * time))
        )
    # FP32 系数只允许舍入级误差，不允许把 ODE 误作为 SDE。
    mean, std = flow_transition(
        latent, velocity, torch.tensor([index]), num_steps=5, noise_level=0.5
    )
    torch.testing.assert_close(mean.flatten(), torch.tensor(expected), atol=2e-7, rtol=2e-6)
    assert float(std) == pytest.approx(math.sqrt(delta) * sigma, rel=2e-6)


def test_gaussian_and_executed_dimensions() -> None:
    # 高斯概率由 torch.distributions 独立核对；未执行时刻和补齐维度不参与 PPO。
    torch.manual_seed(7)
    mean, sample = torch.randn(2, 10, 32), torch.randn(2, 10, 32)
    std = torch.tensor([0.2, 0.4])[:, None, None]
    lengths = torch.tensor([1, 5])
    # 概率必须逐元素一致，块概率必须求和而非取均值。
    logprob = gaussian_logprob(sample, mean, std)
    torch.testing.assert_close(logprob, torch.distributions.Normal(mean, std).log_prob(sample))
    actual = chunk_logprob(logprob, lengths)
    expected = torch.stack([logprob[0, :1, :7].sum(), logprob[1, :5, :7].sum()])
    torch.testing.assert_close(actual, expected)
    # 改动补齐动作和未执行尾部不能改变比例。
    modified = logprob.clone()
    modified[:, :, 7:] = 1234
    modified[0, 1:] = -999
    modified[1, 5:] = 999
    torch.testing.assert_close(chunk_logprob(modified, lengths), actual)


def test_duration_gae_terminal_truncation_and_reset() -> None:
    # 三条并行轨迹分别正常继续、终止和时间截断，最后一行模拟新 episode 的大回报。
    rewards = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [100.0, 100.0, 100.0]])
    values = torch.ones_like(rewards)
    next_values = torch.full_like(rewards, 3.0)
    lengths = torch.tensor([[1, 1, 1], [2, 2, 2], [1, 1, 1]])
    terminated = torch.zeros_like(rewards, dtype=torch.bool)
    truncated = terminated.clone()
    terminated[1, 1] = True
    truncated[1, 2] = True
    # 终止不 bootstrap，截断 bootstrap；两者均阻断新 episode 的优势。
    advantages, returns = duration_gae(
        rewards, values, next_values, lengths, terminated, truncated, gamma=0.5, gae_lambda=0.8
    )
    assert advantages[1, 1].item() == pytest.approx(1.0)
    assert advantages[1, 2].item() == pytest.approx(1.75)
    assert advantages[1, 0].item() == pytest.approx(1.75 + 0.4**2 * 100.5)
    assert advantages[0, 1].item() == pytest.approx(1.5 + 0.4)
    torch.testing.assert_close(returns, advantages + values)


def test_ppo_ratio_one_and_gradient() -> None:
    # 相同新旧策略应为 ratio=1、KL=0；非零优势仍应产生可训练梯度。
    current = torch.tensor([-0.5, -0.8], requires_grad=True)
    old = current.detach().clone()
    advantage = torch.tensor([1.0, -1.0])
    values = torch.zeros(2, requires_grad=True)
    # 梯度方向必须推动高优势动作，同时保持数值有限。
    loss, metrics = ppo_loss(current, old, advantage, values, values.detach(), torch.ones(2))
    loss.backward()
    assert metrics["approx_kl"] == 0 and metrics["clip_fraction"] == 0
    torch.testing.assert_close(current.grad, torch.tensor([-0.5, 0.5]))
    assert torch.isfinite(values.grad).all()


def test_canvas_contact_lift_and_wrong_color() -> None:
    # 无笔尖接触不得出墨，离纸必须断线；同形不同颜色必须判为错误。
    canvas = ColorCanvas(64, 3)
    blank = canvas.image.tobytes()
    canvas.touch(0, (0.1, 0.5), 0)
    assert canvas.image.tobytes() == blank
    # 两段接触之间插入离纸，检查中间像素保持空白。
    canvas.touch(0, (0.1, 0.5), 0.1)
    canvas.touch(0, (0.2, 0.5), 0.1)
    canvas.touch(0, (0.3, 0.5), 0)
    canvas.touch(0, (0.8, 0.5), 0.1)
    assert canvas.image.getpixel((32, 32)) == (255, 255, 255)
    # 目标包含红线，绿色同形不能得到重叠奖励。
    target = render_strokes((Stroke(0, ((0.1, 0.5), (0.8, 0.5))),), 64, 3)
    wrong = render_strokes((Stroke(1, ((0.1, 0.5), (0.8, 0.5))),), 64, 3)
    assert color_iou(target, target)[0] == 1
    assert color_iou(wrong, target)[0] == 0
    assert color_iou(Image.new("RGB", (64, 64), "white"), target)[0] == 0
    # 重置必须清除历史线段端点，防止跨 episode 连线。
    canvas.reset()
    assert canvas.contact_samples == 0 and canvas.last == [None, None, None]
    assert canvas.image.tobytes() == blank


def test_generator_and_one_off_schedule_rewards() -> None:
    # 固定种子复现布局，评测种子不同；抓放里程碑不可重复领取。
    task = generate_task(7, 2)
    assert task == generate_task(7, 2) and task != generate_task(1_000_007, 2)
    assert len({stroke.color for stroke in task.strokes}) == 2
    assert sorted(task.rack_colors) == [0, 1, 2]
    schedule = PaintSchedule(task)
    # 对每种颜色完整走过抓取、绘制、归还，重复抓取标志不能多发奖励。
    reward = 0.0
    for _ in range(2):
        value, changed = schedule.advance(held=True, drawn=False, returned=False)
        assert changed and value == 0.1
        reward += value
        assert schedule.advance(held=True, drawn=False, returned=False) == (0.0, False)
        assert schedule.advance(held=True, drawn=True, returned=False) == (0.0, True)
        value, changed = schedule.advance(held=False, drawn=True, returned=True)
        assert changed and value == 0.1
        reward += value
    assert schedule.phase == "done" and reward == pytest.approx(0.4)


def test_demonstration_alignment_and_phase_mask(tmp_path: Path) -> None:
    # 人工动作序号对应动作前观测序号，换阶段后不能继续作为旧 prompt 的目标。
    episode = tmp_path / "episode-0000"
    episode.mkdir()
    count = 5
    arrays = {
        "actions": np.repeat(np.arange(count)[:, None], 7, axis=1).astype(np.float32),
        "phase": np.array([0, 0, 1, 1, 1]),
        "state": np.zeros((count, 8), np.float32),
        "prompt": np.array(["pick", "pick", "draw", "draw", "draw"]),
    }
    for key in ("image", "wrist_image"):
        arrays[key] = np.broadcast_to(
            np.arange(count, dtype=np.uint8)[:, None, None, None], (count, 8, 8, 3)
        ).copy()
    for key, value in arrays.items():
        np.save(episode / f"{key}.npy", value)
    (episode / "result.json").write_text(json.dumps({"success": True}))
    # 尾部补齐为零，长度只覆盖当前子任务中的真实未来动作。
    data = Demonstrations(str(tmp_path))
    sample, length = data[1]
    assert length == 1 and sample["actions"].shape == (10, 7)
    assert sample["observation/image"][0, 0, 0] == sample["actions"][0, 0] == 1
    assert sample["prompt"] == "pick" and np.count_nonzero(sample["actions"][1:]) == 0
    assert data[2][1] == 3 and data[4][1] == 1


def test_paired_evaluation_excludes_missing_cases() -> None:
    # 未完成三模型配对的 case 不得计作失败，也不能掺进主指标；Long 单独报告。
    rows = [
        {
            "checkpoint": label,
            "suite": suite,
            "task": 0,
            "episode": 0,
            "seed": 7,
            "initial_state_sha256": "same",
            "success": label != "init",
        }
        for label in LABELS
        for suite in ("libero_spatial", "libero_10")
    ]
    rows.append({**rows[0], "task": 1})
    # 已配对 Spatial 和 Long 各一个，另一个 Spatial 留在未配对账目。
    summary = paired_summary(rows)
    assert summary["paired_cases"] == 2 and summary["unpaired_cases"] == 1
    assert summary["groups"]["primary"]["cases"] == 1
    assert summary["groups"]["long"]["cases"] == 1
    assert summary["groups"]["primary"]["paired_deltas"]["rl_minus_init"] == 1
    # 重复记录和初态不一致都应失败，防止虚增统计显著性。
    with pytest.raises(AssertionError, match="duplicate"):
        paired_summary(rows + [rows[0]])
    rows[0] = {**rows[0], "initial_state_sha256": "different"}
    with pytest.raises(AssertionError):
        paired_summary(rows)


def test_explicit_node_placement() -> None:
    # 跨池渲染必须精确绑定指定节点，不能被按地址排序的其他节点抢占。
    nodes = [
        {"Alive": True, "NodeManagerAddress": ip, "NodeID": ip}
        for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3")
    ]
    placement = PlacementSpec(num_nodes=2, node_ips=("10.0.0.3", "10.0.0.2"))
    # 只模拟节点清单，实际选择逻辑仍走 Carrot 的唯一 Ray 集成层。
    with patch("carrot.distributed.ray_runtime.ray.nodes", return_value=nodes):
        selected = RayRuntime._selected_nodes(placement.num_nodes, placement.node_ips)
        assert [n["NodeManagerAddress"] for n in selected] == ["10.0.0.2", "10.0.0.3"]
        with pytest.raises(AssertionError, match="unavailable"):
            RayRuntime._selected_nodes(1, ("10.0.0.4",))
    # 配置中的节点数、去重和非空地址必须一致。
    with pytest.raises(AssertionError):
        PlacementSpec(num_nodes=2, node_ips=("10.0.0.2", "10.0.0.2"))


def test_libero_action_and_state_conventions() -> None:
    # 固定 robosuite OSC 单位、夹爪符号与参考点，防止联合训练把关节命令当 EEF 命令。
    position = np.array([0.4, 0, 0.3])
    rotation = Rotation.identity()
    target, orientation, grip = command_target(
        position, rotation, -1.0, np.array([2.0, -2.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    )
    # 饱和后每轴最多 5cm/0.5rad，夹爪每控制步仅累加 0.01。
    np.testing.assert_allclose(target - position, [0.05, -0.05, 0.05])
    np.testing.assert_allclose(orientation.as_rotvec(), [0, 0, 0.5])
    assert grip == pytest.approx(-0.99)
    np.testing.assert_allclose(EEF_OFFSET, [0, 0, 0.097])
    np.testing.assert_allclose(EEF_ROTATION.apply([1, 0, 0]), [0, -1, 0], atol=1e-15)
    # MuJoCo 的第二手指为负；固定台架平移不泄露任何物体或专家状态。
    state = libero_state(position, rotation, np.array([0.04, 0.03]))
    np.testing.assert_allclose(state[:3], position + LIBERO_BASE, atol=1e-7)
    np.testing.assert_allclose(state[3:], [0, 0, 0, 0.04, -0.03])
