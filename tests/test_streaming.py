from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from tenacity import wait_none

from xps_agent.config import Settings, save_model_settings
from xps_agent.db import StateDB
from xps_agent.llm import SiliconFlowClient, LLMError, LLMTimeoutError, LLMBudgetExceeded
from xps_agent.tasks import OperationStopped


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "fake-stream-key")
    settings = replace(Settings.load(tmp_path), siliconflow_api_key="fake-stream-key")
    settings.ensure_workspace()
    db = StateDB(settings.database_path)
    db.initialize()
    return settings, db


class Stream(httpx.SyncByteStream):
    def __init__(self, frames, *, fail=False):
        self.frames = frames
        self.fail = fail
        self.closed = False

    def __iter__(self):
        for frame in self.frames:
            yield frame
        if self.fail:
            raise httpx.ReadTimeout("private provider error")

    def close(self):
        self.closed = True


def event(delta=None, *, finish=None, usage=None):
    payload = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        payload = {"choices": [], "usage": usage}
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


def response(frames, **kwargs):
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        stream=Stream(frames, **kwargs),
    )


def client(settings, db, handler, **kwargs):
    result = SiliconFlowClient(settings, db, **kwargs)
    result.client.close()
    result.client = httpx.Client(
        base_url=settings.siliconflow_base_url, transport=httpx.MockTransport(handler)
    )
    return result


def test_stream_json_progress_usage_and_cache_keep_legacy_identity(setup):
    settings, db = setup
    seen, progress = [], []

    def handler(request):
        seen.append(request)
        assert json.loads(request.content)["stream"] is True
        return response(
            [
                b": heartbeat\n\n",
                event({"reasoning_content": "private reasoning"}),
                event({"content": '{"ok":'}),
                event({"content": "true}"}, finish="stop"),
                event(usage={"prompt_tokens": 10, "completion_tokens": 5}),
                b"data: [DONE]\n\n",
            ]
        )

    with client(settings, db, handler, progress=lambda **value: progress.append(value)) as llm:
        assert llm.chat_json([{"role": "user", "content": "test"}]) == {"ok": True}
    with client(replace(settings, llm_stream_enabled=False), db, handler) as llm:
        assert llm.chat_json([{"role": "user", "content": "test"}]) == {"ok": True}
        assert llm.network_requests == 0
    assert len(seen) == 1
    assert len(seen[0].headers["X-Trace-Id"]) == 32
    assert any(item.get("stage") == "模型正在推理" for item in progress)
    assert any(item.get("generated_chars") == 11 for item in progress)
    assert "private reasoning" not in json.dumps(progress)
    row = db.rows("SELECT input_tokens,output_tokens FROM llm_calls")[0]
    assert row == {"input_tokens": 10, "output_tokens": 5}


def test_stream_tool_fragments_assemble_without_running_tools(setup):
    settings, db = setup
    frames = [
        event(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_", "arguments": '{"query":'},
                    },
                    {
                        "index": 1,
                        "id": "call_2",
                        "function": {"name": "get_health_report", "arguments": "{"},
                    },
                ]
            }
        ),
        event(
            {
                "tool_calls": [
                    {"index": 0, "function": {"name": "evidence", "arguments": '"XPS"}'}},
                    {"index": 1, "function": {"arguments": "}"}},
                ]
            },
            finish="tool_calls",
        ),
        b"data: [DONE]\n\n",
    ]
    with client(settings, db, lambda request: response(frames)) as llm:
        body = llm.chat(
            [{"role": "user", "content": "test"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
        )
    calls = body["choices"][0]["message"]["tool_calls"]
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"] == {"name": "search_evidence", "arguments": '{"query":"XPS"}'}
    assert calls[1]["function"]["arguments"] == "{}"


@pytest.mark.parametrize("fail", [False, True])
def test_partial_stream_does_not_cache_or_auto_resubmit(setup, fail):
    settings, db = setup
    seen, streams = [], []

    def handler(request):
        seen.append(request)
        result = response([event({"content": '{"partial":'})], fail=fail)
        streams.append(result.stream)
        return result

    with client(settings, db, handler) as llm:
        with pytest.raises(LLMTimeoutError, match="不自动重复提交"):
            llm.chat_json([{"role": "user", "content": "test"}])
    assert len(seen) == 1 and streams[0].closed
    assert db.rows("SELECT COUNT(*) count FROM llm_calls")[0]["count"] == 0
    assert db.rows("SELECT status FROM llm_request_events") == [{"status": "failed"}]


def test_explicit_timeout_retry_is_one_extra_request_only(setup, monkeypatch):
    settings, db = setup
    settings = replace(settings, llm_timeout_retries=1)
    monkeypatch.setattr("xps_agent.llm.wait_exponential_jitter", lambda **kwargs: wait_none())
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ReadTimeout("ambiguous")
        return response([event({"content": '{"ok":true}'}, finish="stop"), b"data: [DONE]\n\n"])

    with client(settings, db, handler) as llm:
        assert llm.chat_json([{"role": "user", "content": "test"}]) == {"ok": True}
    assert len(seen) == 2
    seen.clear()

    def failed(request):
        seen.append(request)
        raise httpx.ReadTimeout("ambiguous")

    with client(settings, db, failed) as llm:
        with pytest.raises(LLMTimeoutError):
            llm.chat([{"role": "user", "content": "new"}])
    assert len(seen) == 2  # Not four billable resubmissions.


def test_timeout_retry_still_respects_action_request_budget(setup, monkeypatch):
    settings, db = setup
    settings = replace(settings, llm_timeout_retries=1, llm_max_requests_per_action=1)
    monkeypatch.setattr("xps_agent.llm.wait_exponential_jitter", lambda **kwargs: wait_none())
    seen = []

    def handler(request):
        seen.append(request)
        raise httpx.ReadTimeout("ambiguous")

    with client(settings, db, handler) as llm:
        with pytest.raises(LLMBudgetExceeded):
            llm.chat([{"role": "user", "content": "test"}])
    assert len(seen) == 1


def test_main_and_vision_can_wait_longer_than_old_caps(setup):
    settings, db = setup
    settings = replace(settings, llm_timeout_seconds=900, vision_timeout_seconds=480)
    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"])
        return response([event({"content": "ok"}, finish="stop"), b"data: [DONE]\n\n"])

    with client(settings, db, handler) as llm:
        llm.chat([{"role": "user", "content": "test"}])
        llm.chat([{"role": "user", "content": "test"}], model=settings.vision_model)
    assert [item["read"] for item in seen] == [900, 480]
    assert all(item["connect"] == 10 and item["write"] == 60 for item in seen)


def test_stream_cancellation_closes_connection_without_caching(setup):
    settings, db = setup
    stop, streams = [], []

    def progress(**value):
        if value.get("stage") == "模型正在生成":
            stop.append(True)

    def check():
        if stop:
            raise OperationStopped("cancelled")

    def handler(request):
        result = response(
            [
                event({"content": "partial"}),
                event({"content": "rest"}, finish="stop"),
                b"data: [DONE]\n\n",
            ]
        )
        streams.append(result.stream)
        return result

    with client(settings, db, handler, progress=progress, check_cancelled=check) as llm:
        with pytest.raises(OperationStopped):
            llm.chat([{"role": "user", "content": "test"}])
    assert streams[0].closed
    assert db.rows("SELECT status FROM llm_request_events") == [{"status": "interrupted"}]
    assert db.rows("SELECT COUNT(*) count FROM llm_calls")[0]["count"] == 0


def test_stream_cooperative_total_deadline(setup, monkeypatch):
    settings, db = setup
    settings = replace(settings, request_timeout_seconds=1)
    ticks = iter(range(0, 100, 2))
    monkeypatch.setattr("xps_agent.llm.time.monotonic", lambda: next(ticks, 100))
    with client(
        settings, db, lambda request: response([event({"content": "ok"}, finish="stop")])
    ) as llm:
        with pytest.raises(LLMTimeoutError, match="总时限"):
            llm.chat([{"role": "user", "content": "test"}])
    assert db.rows("SELECT COUNT(*) count FROM llm_calls")[0]["count"] == 0


def test_malformed_stream_is_not_published(setup):
    settings, db = setup
    with client(settings, db, lambda request: response([b"data: {bad}\n\n"])) as llm:
        with pytest.raises(LLMError, match="valid JSON"):
            llm.chat([{"role": "user", "content": "test"}])
    assert db.rows("SELECT COUNT(*) count FROM llm_calls")[0]["count"] == 0


def test_model_settings_allowlist_bounds_and_preserve_keys(tmp_path):
    (tmp_path / ".env").write_text(
        "SILICONFLOW_API_KEY=fake-preserved-secret\nXPS_STORAGE_ROOT=D:/data/XPS-agent\n",
        encoding="utf-8",
    )
    save_model_settings(
        tmp_path, {"XPS_LLM_TIMEOUT_SECONDS": "600", "XPS_LLM_STREAM_ENABLED": "true"}
    )
    contents = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "fake-preserved-secret" in contents and "XPS_STORAGE_ROOT=D:/data/XPS-agent" in contents
    for updates in (
        {"SILICONFLOW_API_KEY": "not-allowed"},
        {"XPS_LLM_TIMEOUT_SECONDS": "nan"},
        {"XPS_LLM_TIMEOUT_RETRIES": "2"},
        {"XPS_LLM_STREAM_ENABLED": "invalid"},
    ):
        with pytest.raises(ValueError):
            save_model_settings(tmp_path, updates)


def test_model_timeout_settings_can_be_saved_in_web(tmp_path, monkeypatch):
    monkeypatch.setenv("XPS_PROJECT_ROOT", str(tmp_path))
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(
            str(Path(__file__).parents[1] / "src" / "xps_agent" / "dashboard.py"), default_timeout=8
        ).run()
        app.sidebar.radio[0].set_value("系统设置").run()
        next(
            item for item in app.number_input if item.label == "主模型无数据等待上限（秒）"
        ).set_value(900)
        next(item for item in app.button if item.label == "保存模型等待设置").click().run()
        assert not app.exception
        assert any("对下一次任务生效" in item.value for item in app.success)
        assert Settings.load(tmp_path).llm_timeout_seconds == 900
        assert Settings.load(tmp_path).llm_timeout_retries == 0
    finally:
        st.cache_resource.clear()
