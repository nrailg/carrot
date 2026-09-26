import threading
from contextlib import contextmanager

import numpy as np
import pytest
from openpi_client import msgpack_numpy
from websockets.sync.server import serve

from examples.so101_real.client import PolicyClient


@contextmanager
def policy_server(handler):
    server = serve(handler, "127.0.0.1", 0, compression=None, close_timeout=0.1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_real_websocket_numpy_roundtrip():
    # 使用真实本地 WebSocket 验证二进制协议，不能只 mock infer 方法。
    expected = np.arange(24, dtype=np.float32).reshape(4, 6)
    received = []

    def handler(socket):
        socket.send(msgpack_numpy.packb({"embodiment": "so101", "action_dim": 6}))
        received.append(msgpack_numpy.unpackb(socket.recv()))
        socket.send(msgpack_numpy.packb({"actions": expected}))

    # 图像和状态应无损往返，客户端不得重命名或归一化字段。
    observation = {"observation/state": np.arange(6, dtype=np.float32),
                   "observation/image": np.zeros((12, 16, 3), dtype=np.uint8)}
    with policy_server(handler) as uri:
        client = PolicyClient(uri, 2, 2)
        try:
            response = client.infer(observation)
            assert client.metadata["embodiment"] == "so101"
        finally:
            client.close()

    # 明确验证 dtype 和逐元素数值，避免列表隐式转换掩盖协议问题。
    assert response["actions"].dtype == np.float32
    np.testing.assert_array_equal(response["actions"], expected)
    np.testing.assert_array_equal(
        received[0]["observation/state"], observation["observation/state"]
    )


def test_receive_timeout_is_bounded():
    # 服务端收到请求却不返回时，客户端必须在有限时间失败，不能无限等待。
    release = threading.Event()

    def handler(socket):
        socket.send(msgpack_numpy.packb({}))
        socket.recv()
        release.wait(timeout=2)

    # 故意阻塞响应，测试真实 recv timeout 并确保释放测试服务器线程。
    with policy_server(handler) as uri:
        client = PolicyClient(uri, 1, 0.02)
        try:
            with pytest.raises(TimeoutError):
                client.infer({})
        finally:
            release.set()
            client.close()


def test_text_server_error_propagates():
    # OpenPI 服务端用文本返回异常，客户端必须保留错误信息而非尝试执行。
    def handler(socket):
        socket.send(msgpack_numpy.packb({}))
        socket.recv()
        socket.send("invalid observation")

    # 通过真实网络接收异常帧，验证失败语义。
    with policy_server(handler) as uri:
        client = PolicyClient(uri, 1, 1)
        try:
            with pytest.raises(RuntimeError, match="invalid observation"):
                client.infer({})
        finally:
            client.close()
