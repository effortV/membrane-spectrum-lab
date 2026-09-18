"""Cloud-only SSH/HTTP client: no local database, filesystem data or provider API.

HTTP uses a direct-tcpip channel inside SSH; no public API or local proxy port.
Host identity is pinned BEFORE authentication. Writes are never auto-retried.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import hmac
import http.client
from io import BufferedReader, RawIOBase, StringIO
import json
import re
import socket
import threading
from urllib.parse import urlencode

import paramiko


class BackendError(RuntimeError):
    pass


class SubmissionUncertain(BackendError):
    pass


@dataclass(frozen=True)
class ConnectionSettings:
    host: str
    port: int
    username: str
    private_key: str = field(repr=False)
    host_key_sha256: str = ""
    backend_token: str = field(default="", repr=False)
    socks_host: str = ""
    socks_port: int = 1055

    @classmethod
    def from_mapping(cls, values):
        required = (
            "XPS_SSH_HOST",
            "XPS_SSH_PRIVATE_KEY",
            "XPS_SSH_HOST_KEY_SHA256",
            "XPS_BACKEND_TOKEN",
        )
        if any(not str(values.get(key, "")).strip() for key in required):
            raise BackendError("尚未配置服务器连接凭据；不会在云端创建替代数据库。")
        try:
            result = cls(
                str(values["XPS_SSH_HOST"]).strip(),
                int(values.get("XPS_SSH_PORT", 2222)),
                str(values.get("XPS_SSH_USERNAME", "xpscloud")),
                str(values["XPS_SSH_PRIVATE_KEY"]),
                str(values["XPS_SSH_HOST_KEY_SHA256"]).strip(),
                str(values["XPS_BACKEND_TOKEN"]).strip(),
                str(values.get("XPS_SSH_SOCKS_HOST", "")).strip(),
                int(values.get("XPS_SSH_SOCKS_PORT", 1055)),
            )
        except (TypeError, ValueError):
            raise BackendError("服务器连接配置格式不正确。") from None
        if (
            not 1 <= result.port <= 65535
            or result.username != "xpscloud"
            or not re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", result.host_key_sha256)
            or len(result.backend_token) < 32
        ):
            raise BackendError("服务器端口、专用账号、指纹或接口凭据不正确。")
        if result.socks_host and (
            result.socks_host != "127.0.0.1" or not 1 <= result.socks_port <= 65535
        ):
            raise BackendError("网络代理必须仅监听云端回环地址。")
        return result


class ChannelReader(RawIOBase):
    def __init__(self, socket):
        super().__init__()
        self.socket = socket

    def readable(self):
        return True

    def readinto(self, buffer):
        content = self.socket.channel.recv(len(buffer))
        buffer[: len(content)] = content
        return len(content)

    def close(self):
        if not self.closed:
            super().close()
            self.socket.file_closed()


class ChannelSocket:
    """socket.makefile lifecycle: closing socket must not discard response body.

    Unlike real sockets, Paramiko Channel.close immediately kills ChannelFile reads.
    HTTPConnection closes its socket as soon as Connection: close headers arrive,
    BEFORE reading the body. Retain the channel until the response file is closed.
    """

    def __init__(self, channel):
        self.channel = channel
        self.files = 0
        self.closed = False
        self.lock = threading.RLock()

    def makefile(self, mode):
        if mode != "rb":
            raise ValueError("Only binary HTTP response reading is supported.")
        with self.lock:
            self.files += 1
        return BufferedReader(ChannelReader(self))

    def sendall(self, content):
        self.channel.sendall(content)

    def settimeout(self, value):
        self.channel.settimeout(value)

    def close(self):
        with self.lock:
            self.closed = True
            if not self.files:
                self.channel.close()

    def file_closed(self):
        with self.lock:
            self.files -= 1
            if self.closed and not self.files:
                self.channel.close()


class ChannelHTTPConnection(http.client.HTTPConnection):
    def __init__(self, transport, timeout):
        super().__init__("127.0.0.1", 8771, timeout=timeout)
        self.transport = transport

    def connect(self):
        self.sock = ChannelSocket(
            self.transport.open_channel(
                "direct-tcpip", ("127.0.0.1", 8771), ("127.0.0.1", 0), timeout=self.timeout
            )
        )
        self.sock.settimeout(self.timeout)


class SSHBackendClient:
    def __init__(self, settings: ConnectionSettings):
        self.settings = settings
        self._transport = None
        self._lock = threading.RLock()

    def close(self):
        with self._lock:
            if self._transport is not None:
                self._transport.close()
                self._transport = None

    def _connect(self):
        with self._lock:
            if (
                self._transport is not None
                and self._transport.is_active()
                and self._transport.is_authenticated()
            ):
                return self._transport
            self.close()
            connection = None
            transport = None
            try:
                if self.settings.socks_host:
                    import socks

                    connection = socks.socksocket()
                    connection.set_proxy(
                        socks.SOCKS5, self.settings.socks_host, self.settings.socks_port, rdns=True
                    )
                    connection.settimeout(20)
                    connection.connect((self.settings.host, self.settings.port))
                else:
                    connection = socket.create_connection(
                        (self.settings.host, self.settings.port), timeout=20
                    )
                transport = paramiko.Transport(connection)
                transport.banner_timeout = 20
                transport.auth_timeout = 30
                transport.start_client(timeout=20)
                observed = "SHA256:" + base64.b64encode(
                    hashlib.sha256(transport.get_remote_server_key().asbytes()).digest()
                ).decode().rstrip("=")
                if not hmac.compare_digest(observed, self.settings.host_key_sha256):
                    raise BackendError("SSH 主机指纹不匹配，已拒绝认证；请核实服务器身份。")
                key = None
                for key_type in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
                    try:
                        key = key_type.from_private_key(StringIO(self.settings.private_key))
                        break
                    except (paramiko.SSHException, ValueError):
                        continue
                if key is None:
                    raise BackendError("专用 SSH 私钥无法读取；不要使用管理员私钥。")
                transport.auth_publickey(self.settings.username, key)
                if not transport.is_authenticated():
                    raise BackendError("服务器未接受专用 SSH 身份。")
                transport.set_keepalive(30)
                self._transport = transport
                return transport
            except Exception as error:
                if transport is not None:
                    transport.close()
                elif connection is not None:
                    connection.close()
                if isinstance(error, BackendError):
                    raise
                raise BackendError(
                    "无法建立服务器 SSH 连接，请检查网络、专用密钥和服务器状态。"
                ) from None

    def request(self, method, path, owner, *, body=None, query=None, binary=False):
        if not re.fullmatch(r"[a-f0-9]{32}", owner):
            raise BackendError("会话标识无效。")
        if not path.startswith(("/v1/", "/health")) or "\r" in path or "\n" in path:
            raise BackendError("接口路径不正确。")
        encoded = json.dumps(body, ensure_ascii=True).encode() if body is not None else None
        headers = {
            "Authorization": "Bearer " + self.settings.backend_token,
            "X-XPS-Owner": owner,
            "Accept": "application/json",
            "Connection": "close",
        }
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        # Connecting is before submission; failures here have sent no HTTP write.
        transport = self._connect()
        connection = ChannelHTTPConnection(transport, timeout=45)
        try:
            connection.request(
                method,
                path + ("?" + urlencode(query, doseq=True) if query else ""),
                body=encoded,
                headers=headers,
            )
            response = connection.getresponse()
            expected_length = response.length
            content = response.read(40 * 1024 * 1024 + 1)
            if len(content) > 40 * 1024 * 1024:
                raise BackendError("服务器返回内容过大，请分批读取。")
            if expected_length is not None and len(content) != expected_length:
                raise http.client.IncompleteRead(content, expected_length - len(content))
            if response.status >= 400:
                try:
                    detail = json.loads(content).get("detail", "服务器拒绝请求。")
                except (ValueError, AttributeError):
                    detail = "服务器拒绝请求。"
                if not isinstance(detail, str):
                    detail = "请求字段不正确。"
                raise BackendError(str(detail)[:2000])
            return content if binary else json.loads(content)
        except BackendError:
            raise
        except Exception:
            self.close()
            if method != "GET":
                raise SubmissionUncertain(
                    "提交结果暂不确定，未自动重发。请刷新后台状态，并用原请求编号查询。"
                ) from None
            raise BackendError("读取服务器失败，请手动刷新；未创建云端数据库。") from None
        finally:
            connection.close()
