import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("backend_client", ROOT / "backend_client.py")
module = importlib.util.module_from_spec(spec)
sys.modules["backend_client"] = module
spec.loader.exec_module(module)


def settings(**updates):
    values = {
        "XPS_SSH_HOST": "example.invalid",
        "XPS_SSH_PRIVATE_KEY": "test_private_material",
        "XPS_SSH_HOST_KEY_SHA256": "SHA256:" + "A" * 43,
        "XPS_BACKEND_TOKEN": "test_token_" + "x" * 40,
    }
    values.update(updates)
    return module.ConnectionSettings.from_mapping(values)


def test_missing_config_never_falls_back_to_cloud_database():
    with pytest.raises(module.BackendError, match="替代数据库"):
        module.ConnectionSettings.from_mapping({})
    value = settings()
    assert value.private_key not in repr(value) and value.backend_token not in repr(value)
    with pytest.raises(module.BackendError):
        settings(XPS_SSH_USERNAME="ASUS")
    with pytest.raises(module.BackendError):
        settings(XPS_SSH_SOCKS_HOST="0.0.0.0")


def test_host_fingerprint_is_checked_before_authentication(monkeypatch):
    events = []

    class Key:
        def asbytes(self):
            return b"server_key"

    class Transport:
        def __init__(self, sock):
            pass

        def start_client(self, **kwargs):
            events.append("handshake")

        def get_remote_server_key(self):
            events.append("host_key")
            return Key()

        def auth_publickey(self, *args):
            events.append("authenticate")

        def close(self):
            events.append("close")

    monkeypatch.setattr(module.socket, "create_connection", lambda *a, **k: object())
    monkeypatch.setattr(module.paramiko, "Transport", Transport)
    client = module.SSHBackendClient(settings())
    with pytest.raises(module.BackendError, match="指纹不匹配"):
        client._connect()
    assert events == ["handshake", "host_key", "close"]


def test_write_timeout_does_not_repeat_submission(monkeypatch):
    attempts = []

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            attempts.append(True)
            raise TimeoutError("secret details")

        def close(self):
            pass

    monkeypatch.setattr(module, "ChannelHTTPConnection", Connection)
    client = module.SSHBackendClient(settings())
    monkeypatch.setattr(client, "_connect", lambda: object())
    with pytest.raises(module.SubmissionUncertain, match="未自动重发") as error:
        client.request("POST", "/v1/jobs", "a" * 32, body={"kind": "index"})
    assert len(attempts) == 1 and "secret details" not in str(error.value)


def test_forwarding_destination_is_fixed_and_cloud_has_no_backend_startup():
    source = (ROOT / "streamlit_cloud.py").read_text(encoding="utf-8")
    for forbidden in (
        "StateDB",
        "Settings.load",
        "xps_agent.dashboard",
        "xps_agent.backend",
        "SiliconFlowClient",
        "sqlite3",
    ):
        assert forbidden not in source
    source = (ROOT / "backend_client.py").read_text(encoding="utf-8")
    assert '("127.0.0.1", 8771)' in source
    assert "AutoAddPolicy" not in source


def test_connection_close_headers_do_not_discard_later_ssh_body():
    body = b'{"status":"ok"}'

    class Channel:
        def __init__(self):
            self.closed = False
            self.packets = [
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 15\r\n\r\n",
                body,
            ]

        def recv(self, size):
            if self.closed:
                return b""
            return self.packets.pop(0) if self.packets else b""

        def settimeout(self, value):
            pass

        def sendall(self, content):
            pass

        def close(self):
            self.closed = True

    channel = Channel()

    class Transport:
        def open_channel(self, *args, **kwargs):
            return channel

    connection = module.ChannelHTTPConnection(Transport(), timeout=5)
    connection.request("GET", "/health")
    response = connection.getresponse()
    assert not channel.closed
    assert response.read(1000) == body
    assert channel.closed
    connection.close()
