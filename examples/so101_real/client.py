from typing import Any

from openpi_client import msgpack_numpy
from websockets.sync.client import connect


class PolicyClient:
    """Use the OpenPI binary protocol with bounded receives and no reconnect.

    Parameters
    ----------
    uri : str
    connect_timeout : float
    request_timeout : float
    """

    def __init__(self, uri: str, connect_timeout: float, request_timeout: float) -> None:
        self.request_timeout = request_timeout
        self.socket = connect(uri, compression=None, max_size=None,
                              open_timeout=connect_timeout, close_timeout=1)
        try:
            # 握手首先接收服务端元数据；失败时立即释放已建立的连接。
            self.metadata = self._receive(connect_timeout)
        except BaseException:
            self.socket.close()
            raise
        self.packer = msgpack_numpy.Packer()

    def _receive(self, timeout: float) -> dict[str, Any]:
        response = self.socket.recv(timeout=timeout)
        # OpenPI 服务端以文本帧传递错误，正常结果是二进制 msgpack。
        if isinstance(response, str):
            raise RuntimeError(f"policy server error: {response}")
        result = msgpack_numpy.unpackb(response)
        assert isinstance(result, dict), "policy response must be a mapping"
        return result

    def infer(self, observation: dict[str, Any], *, timeout: float | None = None) -> dict:
        self.socket.send(self.packer.pack(observation))
        return self._receive(self.request_timeout if timeout is None else timeout)

    def close(self) -> None:
        self.socket.close()
