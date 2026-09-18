import base64
from pathlib import Path
import threading
import time
import uuid

from fastapi.testclient import TestClient
import pytest

from xps_agent.backend import create_app, start_embedded_api
from xps_agent.config import Settings
from xps_agent.conversations import ConversationStore, RUNTIME_ID
from xps_agent.db import StateDB
from xps_agent.tasks import BackgroundTasks

TOKEN = "test_backend_token_" + "x" * 40
OWNER = "a" * 32


@pytest.fixture
def backend(tmp_path):
    settings = Settings.load(tmp_path)
    settings.ensure_workspace()
    db = StateDB(settings.database_path)
    db.initialize()
    manager = BackgroundTasks(threading.Lock(), {})
    client = TestClient(create_app(settings, db, manager, TOKEN))
    client.headers.update({"Authorization": "Bearer " + TOKEN, "X-XPS-Owner": OWNER})
    return client, settings, db, manager


def test_backend_has_no_anonymous_routes_or_docs(backend):
    client, settings, db, manager = backend
    assert client.get("/health", headers={"Authorization": ""}).status_code == 401
    assert client.get("/health", headers={"X-XPS-Owner": "invalid"}).status_code == 400
    assert client.get("/health").json()["database_location"] == "server_only"
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/v1/overview").json()["counts"]["documents"] == 0
    assert db.rows("SELECT COUNT(*) n FROM llm_request_events")[0]["n"] == 0


def test_backend_token_required_and_opt_in(backend):
    client, settings, db, manager = backend
    with pytest.raises(ValueError):
        create_app(settings, db, manager, "weak")
    assert start_embedded_api(settings, db, manager) is None


def test_backend_validation_does_not_echo_payload_or_accept_paths(backend):
    client, *_ = backend
    secret = "never_echo_this_value"
    invalid = client.post("/v1/jobs", json={"request_id": secret, "kind": "index"})
    assert invalid.status_code == 422 and secret not in invalid.text
    invalid = client.post(
        "/v1/jobs",
        json={"request_id": uuid.uuid4().hex, "kind": "parse", "parameters": {"path": secret}},
    )
    assert invalid.status_code == 422 and secret not in invalid.text
    invalid = client.post("/v1/jobs", json={"request_id": uuid.uuid4().hex, "kind": "shell"})
    assert invalid.status_code == 400


def test_duplicate_job_starts_one_worker_and_checks_payload(backend, monkeypatch):
    client, settings, db, manager = backend
    entered, release = threading.Event(), threading.Event()
    calls = []

    def build(*args):
        def worker(context):
            calls.append(True)
            entered.set()
            release.wait(3)
            return {"documents": 1}

        return worker

    monkeypatch.setattr("xps_agent.backend.build_worker", build)
    body = {"request_id": uuid.uuid4().hex, "kind": "index", "parameters": {}}
    try:
        first = client.post("/v1/jobs", json=body)
        assert first.status_code == 202 and entered.wait(1)
        duplicate = client.post("/v1/jobs", json=body)
        assert duplicate.json()["duplicate"]
        assert duplicate.json()["task_id"] == first.json()["task_id"] and len(calls) == 1
        body["kind"] = "plan"
        assert client.post("/v1/jobs", json=body).status_code == 409
        assert client.get("/v1/tasks").json()["current"]["is_owner"]
        assert (
            client.post(
                "/v1/tasks/" + first.json()["task_id"] + "/cancel",
                headers={"X-XPS-Owner": "b" * 32},
            ).status_code
            == 400
        )
        assert client.post("/v1/tasks/" + first.json()["task_id"] + "/cancel").status_code == 200
    finally:
        release.set()
        deadline = time.monotonic() + 3
        while manager.paid_lock.locked() and time.monotonic() < deadline:
            time.sleep(0.01)


def test_failed_submission_is_not_automatically_retried(backend, monkeypatch):
    client, settings, db, manager = backend
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("busy")

    monkeypatch.setattr(manager, "start", fail)
    body = {"request_id": uuid.uuid4().hex, "kind": "index"}
    assert client.post("/v1/jobs", json=body).status_code == 409
    second = client.post("/v1/jobs", json=body)
    assert second.json()["duplicate"] and second.json()["status"] == "failed"
    assert len(calls) == 1
    assert client.get("/v1/requests/" + body["request_id"]).json()["status"] == "failed"


def test_paid_jobs_require_confirmation_and_server_key(backend):
    client, settings, db, manager = backend
    for kind in ("proposal", "extraction"):
        body = {"request_id": uuid.uuid4().hex, "kind": kind}
        assert client.post("/v1/jobs", json=body).status_code == 400
    body = {"request_id": uuid.uuid4().hex, "kind": "proposal", "confirmed_paid": True}
    assert client.post("/v1/jobs", json=body).status_code == 409
    assert db.rows("SELECT COUNT(*) n FROM llm_request_events")[0]["n"] == 0


def test_conversation_keeps_same_process_context_and_normalized_answers(backend):
    client, settings, db, manager = backend
    key = client.post("/v1/conversations", json={"title": "testing", "memory": "remember"}).json()[
        "conversation_id"
    ]
    store = ConversationStore(db)
    turn = store.reserve(key, "question", model="fake", task_id="t")
    assert db.rows("SELECT runtime_id FROM chat_turns")[0]["runtime_id"] == RUNTIME_ID
    result = client.get("/v1/conversations/" + key)
    assert result.json()["turns"][0]["status"] == "running"
    store.finish(turn, r"\[R = \frac{S_N}{S_O + \epsilon}\]")
    assert "$$" in client.get("/v1/conversations/" + key).json()["turns"][0]["answer"]
    assert "remember" in client.get("/v1/conversations/" + key + "/markdown").text


def test_pdf_upload_stays_inside_server_inbox_and_blocks_file_escape(backend, tmp_path):
    client, settings, db, manager = backend
    content = b"%PDF-1.4\nminimal"
    result = client.post(
        "/v1/uploads",
        json={
            "kind": "pdf",
            "name": "../../example.pdf",
            "content_base64": base64.b64encode(content).decode(),
        },
    )
    assert result.status_code == 200
    row = db.rows("SELECT * FROM documents")[0]
    assert Path(row["local_path"]).resolve().is_relative_to(settings.workspace_root / "inbox")
    assert client.get("/v1/library/pdf", params={"doc_key": row["doc_key"]}).content == content
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4\nconfidential")
    db.upsert_document({"doc_key": "escape", "source": "fake", "local_path": str(outside)})
    assert client.get("/v1/library/pdf", params={"doc_key": "escape"}).status_code == 400
    invalid = client.post(
        "/v1/uploads", json={"kind": "pdf", "name": "bad.pdf", "content_base64": "!!"}
    )
    assert invalid.status_code == 400


def test_body_size_and_maintenance_hold(backend, monkeypatch):
    client, settings, db, manager = backend
    monkeypatch.setattr("xps_agent.backend.MAX_BODY", 100)
    assert client.post("/v1/conversations", json={"title": "x" * 150}).status_code == 413
    (settings.workspace_root / "state" / "migration_hold.json").write_text("{}")
    assert client.get("/health").status_code == 200
    assert client.get("/v1/conversations").status_code == 503


def test_library_batches_identify_added_documents(backend):
    client, settings, db, manager = backend
    from xps_agent.library_activity import LibraryActivity

    db.upsert_document({"doc_key": "doi:1", "title": "new paper", "source": "fake"})
    activity = LibraryActivity(db)
    key = activity.start("search", "XPS")
    activity.item(key, "doi:1", 0, is_new=True)
    activity.finish(key)
    assert client.get("/v1/library/batches/" + key).json()["summary"]["added"] == 1
    assert client.get("/v1/library").json()[0]["title"] == "new paper"


def test_runtime_is_shared_and_api_starts_before_browser(backend, monkeypatch):
    import http.client
    import socket
    from xps_agent.server_runtime import get_runtime

    client, settings, db, manager = backend
    runtime = get_runtime(settings.project_root)
    assert runtime is get_runtime(settings.project_root)
    assert runtime.manager.paid_lock is runtime.paid_lock
    path = settings.workspace_root / "state" / "connections" / "backend_token.txt"
    path.parent.mkdir(parents=True)
    path.write_text(TOKEN, encoding="utf-8")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    monkeypatch.setattr("xps_agent.backend._open_listener", lambda: listener)
    api = runtime.start_api()
    try:
        assert api is runtime.start_api() and api.thread.is_alive()
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "GET", "/health", headers={"Authorization": "Bearer " + TOKEN, "X-XPS-Owner": OWNER}
        )
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        connection.close()
    finally:
        api.server.should_exit = True
        api.thread.join(timeout=3)
