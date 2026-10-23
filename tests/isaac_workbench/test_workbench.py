import json
import multiprocessing
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/isaac_workbench"))

from guest_task import grade  # noqa: E402
from logic import KeySwitches, PaintCanvas, keyboard_layout  # noqa: E402
from tasks import CODING_TASKS  # noqa: E402
from vm import Qmp, VmPool  # noqa: E402


def test_key_edges_require_contact_and_hysteresis():
    # 防止悬空、阈值抖动或长按产生重复按键；释放必须与按下成对。
    switches = KeySwitches(2)
    # 位移相同但没有接触时，不能注入 guest 输入。
    assert switches.update(np.array([0.002, 0]), np.zeros(2)) == []
    assert switches.update(np.array([0.002, 0]), np.array([0.2, 0])) == [(0, True)]
    # 按下之后回弹到迟滞区应保持，真正回到释放阈值才发 key-up。
    assert switches.update(np.array([0.001, 0]), np.array([0.1, 0])) == []
    assert switches.update(np.array([0.0005, 0]), np.zeros(2)) == [(0, False)]
    # reset 清理尚未释放的按键，防止下一回合继承修饰键状态。
    switches.update(np.array([0, 0.002]), np.array([0, 0.2]))
    assert switches.reset() == [(1, False)]
    assert not switches.down.any()


def test_canvas_requires_contact_and_preserves_pen_lifts():
    # 抬笔必须断开两段轨迹；没有接触时画布必须完全不变。
    canvas = PaintCanvas(size=64, width=3)
    blank = np.asarray(canvas.image).copy()
    assert not canvas.touch((0.1, 0.1), 0)
    np.testing.assert_array_equal(np.asarray(canvas.image), blank)
    # 两个有接触的点之间插入无接触点，中心不能被错误连线。
    canvas.touch((0.1, 0.1), 1)
    canvas.touch((0.5, 0.5), 0)
    canvas.touch((0.9, 0.9), 1)
    assert tuple(np.asarray(canvas.image)[32, 32]) == (255, 255, 255)
    assert canvas.contact_samples == 2
    # 越界也应断笔，reset 必须同时清空像素与接触计数。
    canvas.touch((-0.1, 0.5), 1)
    assert canvas.last is None
    canvas.reset()
    np.testing.assert_array_equal(np.asarray(canvas.image), blank)
    assert canvas.contact_samples == 0


def test_keyboard_codes_and_default_program():
    # 符号键使用明确的修饰键组合；每道题的示范必须能由物理键盘输入。
    keys = keyboard_layout()
    labels = {key.label for key in keys}
    assert len({(key.qcode, key.modifiers) for key in keys}) == len(keys)
    for task in CODING_TASKS.values():
        assert set(task.demonstration.strip()) <= labels
    assert "ENTER" in labels


@pytest.mark.parametrize("task", list(CODING_TASKS.values()), ids=list(CODING_TASKS))
def test_exercise_grading(task):
    # 正确代码必须通过全部边界用例，固定答案不能冒充通用解法。
    spec = task.guest_spec("test-episode")
    assert grade(task.demonstration.strip(), spec) == len(task.cases)
    assert grade("0", spec) < len(task.cases)
    assert "demonstration" not in spec
    # 禁止提交代码向宿主通信或伪造评分输出，评分由 guest 的独立通道产生。
    for expression in ("print('WORKBENCH_GRADE')", "__import__('os')", "a.__class__"):
        with pytest.raises(AssertionError):
            grade(expression, spec)


@pytest.mark.skipif(not os.environ.get("WORKBENCH_GUEST"), reason="requires a real Linux guest")
def test_parallel_guests_and_partial_reset(tmp_path):
    # 真正并行启动三个 QEMU，分别做不同的题，验证输入、评分和 reset 隔离。
    keys = keyboard_layout()
    mapping = {key.label: i for i, key in enumerate(keys)}
    mapping["\n"] = mapping["ENTER"]
    tasks = list(CODING_TASKS.values())
    pool = VmPool(Path(os.environ["WORKBENCH_GUEST"]), tmp_path / "vms", tasks)
    guests = []
    try:
        guests = list(pool.guests)
        assert len({guest.process.pid for guest in guests}) == len(tasks)
        assert len({guest.runtime.name for guest in guests}) == len(tasks)
        assert len({guest.episode for guest in guests}) == len(tasks)

        # 先给一个环境错误解，再并行给各环境正确解，错误不能污染其他环境。
        def events(program):
            return [
                (2 * j + int(not down), mapping[c], down)
                for j, c in enumerate(program)
                for down in (True, False)
            ]

        grades = pool.step([events("0\n"), [], []], keys)
        assert not grades[0]["success"] and grades[0]["attempt"] == 1
        assert all(grade["attempt"] == 0 for grade in grades[1:])
        grades = pool.step([events(task.demonstration) for task in tasks], keys)
        assert all(
            grade["success"] and grade["task"] == task.name
            for grade, task in zip(grades, tasks, strict=True)
        )
        assert all(not guest.down for guest in guests)
        # 保留按住的修饰键后只重置 env 0；其余 VM 的进程、屏幕与评分必须完全不变。
        pool.step([[(100, mapping["SHIFT"], True)], [], []], keys)
        untouched = [
            (guest.process.pid, guest.episode, guest.screenshot().tobytes(), guest.grade())
            for guest in guests[1:]
        ]
        pool.reset([0])
        assert guests[0].process.poll() is not None
        assert pool.guests[0].episode != guests[0].episode
        assert pool.guests[0].grade()["attempt"] == 0
        assert not pool.guests[0].down and not pool.pressed[0]
        assert [
            (guest.process.pid, guest.episode, guest.screenshot().tobytes(), guest.grade())
            for guest in pool.guests[1:]
        ] == untouched
        assert pool.step([events(tasks[0].demonstration), [], []], keys)[0]["success"]
        # 单个 VM 崩溃必须让整个 batch 报错，不能把旧分数当作成功返回。
        pool.guests[1].process.kill()
        pool.guests[1].process.wait(timeout=5)
        with pytest.raises(ExceptionGroup):
            pool.step([[], [(999, mapping["a"], True)], []], keys)
    finally:
        remaining = list(pool.guests)
        pool.close()
    # 退出必须回收自己创建的所有 VM，避免后续 RL worker 继承孤儿进程。
    assert all(guest.process.poll() is not None for guest in guests + remaining)


def _spawned_vm_worker(guest_dir: str, output: str, barrier, queue) -> None:
    pool = VmPool(Path(guest_dir), Path(output), list(CODING_TASKS.values())[:2])
    try:
        barrier.wait(timeout=30)
        queue.put([(guest.process.pid, guest.episode, guest.runtime.name) for guest in pool.guests])
        pool.reset([0])
        assert all(guest.grade()["attempt"] == 0 for guest in pool.guests)
    finally:
        pool.close()
    assert all(guest.process.poll() is not None for guest in pool.guests)


@pytest.mark.skipif(not os.environ.get("WORKBENCH_GUEST"), reason="requires a real Linux guest")
def test_multiple_spawned_rl_workers(tmp_path):
    # 两个独立 RL worker 同时启动各自的两个 VM，不能复用进程、episode 或 Unix socket。
    context = multiprocessing.get_context("spawn")
    queue, barrier = context.Queue(), context.Barrier(2)
    workers = [
        context.Process(
            target=_spawned_vm_worker,
            args=(os.environ["WORKBENCH_GUEST"], str(tmp_path / f"worker-{i}"), barrier, queue),
        )
        for i in range(2)
    ]
    for worker in workers:
        worker.start()
    try:
        # barrier 确保四个 VM 同时存活；退出后检查子进程确实成功完成清理。
        records = queue.get(timeout=60) + queue.get(timeout=60)
        assert all(len({record[column] for record in records}) == 4 for column in range(3))
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        queue.close()


def _crashing_vm_worker(guest_dir: str, output: str, queue) -> None:
    pool = VmPool(Path(guest_dir), Path(output), [CODING_TASKS["sum_two"]])
    queue.put(pool.guests[0].process.pid)
    queue.close()
    queue.join_thread()
    # 模拟训练 worker 的不可恢复退出，故意不执行 Python finally。
    os._exit(23)


@pytest.mark.skipif(not os.environ.get("WORKBENCH_GUEST"), reason="requires a real Linux guest")
def test_owner_crash_stops_qemu(tmp_path):
    # 训练 worker 突然退出也必须终止 QEMU，不能留下占用内存的孤儿 VM。
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    worker = context.Process(
        target=_crashing_vm_worker,
        args=(os.environ["WORKBENCH_GUEST"], str(tmp_path / "crash"), queue),
    )
    worker.start()
    try:
        pid = queue.get(timeout=60)
        worker.join(timeout=10)
        assert worker.exitcode == 23
        # PID 1 可能延迟回收 zombie；它已不运行且释放 VM 内存，不误报为存活。
        status = Path(f"/proc/{pid}/status")
        deadline = time.monotonic() + 5
        while status.exists():
            try:
                contents = status.read_text()
            except FileNotFoundError:
                break  # PID 1 在 exists 与 read 之间回收进程也是预期结果。
            state = next(line for line in contents.splitlines() if line.startswith("State:"))
            if "Z (zombie)" in state:
                break
            assert time.monotonic() < deadline, f"orphan QEMU is still running: {pid}"
            time.sleep(0.05)
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        queue.close()


def test_qmp_events_are_not_replies(tmp_path):
    # QMP 会在同步回复间插入事件；不能把 STOP 等事件误当成功回复。
    path = tmp_path / "qmp.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    errors = []

    # 用真实 Unix socket 模拟一次握手、事件夹杂与命令失败。
    def serve():
        try:
            connection, _ = server.accept()
            with connection, connection.makefile("rwb") as stream:
                stream.write(b'{"QMP": {}}\n')
                stream.flush()
                for number in range(3):
                    request = json.loads(stream.readline())
                    stream.write(b'{"event": "STOP"}\n')
                    response = {"id": request["id"]}
                    response.update(
                        {"error": {"desc": "test failure"}}
                        if number == 2
                        else {"return": {"ok": number}}
                    )
                    stream.write(json.dumps(response).encode() + b"\n")
                    stream.flush()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    client = Qmp(path)
    try:
        # 核对回复 id 与返回值，错误必须传播，不能降级为空结果。
        assert client.execute("query-status") == {"ok": 1}
        with pytest.raises(AssertionError, match="test failure"):
            client.execute("bad-command")
    finally:
        client.close()
        worker.join(timeout=3)
        server.close()
    assert not worker.is_alive() and not errors
