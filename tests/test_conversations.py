import json
import threading
import time
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from xps_agent.agent import ResearchAgent
from xps_agent.config import Settings
from xps_agent.conversations import ConversationStore, start_conversation_task
from xps_agent.db import StateDB
from xps_agent.llm import SiliconFlowClient
from xps_agent.tasks import BackgroundTasks, OperationStopped


@pytest.fixture
def db(tmp_path):
    database = StateDB(tmp_path / "state.sqlite3")
    database.initialize()
    return database


def completed(store, key, question="问题", answer="完整回答 [evidence:one]"):
    turn = store.reserve(key, question, model="test-model", task_id="test-task")
    store.finish(turn, answer)
    return turn


def wait_task(manager, owner="test-owner"):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(owner)
        if snapshot and snapshot["status"] != "running":
            # The gate is released just after the status is set.
            if not manager.paid_lock.locked():
                return snapshot
        threading.Event().wait(0.01)
    raise AssertionError("Test task did not finish")


def test_durable_questions_answers_context_and_conversation_isolation(db):
    store = ConversationStore(db)
    first, second = store.create(), store.create("另一研究")
    completed(store, first, "只研究 NF", "引用 DOI / PDF 第 3 页 / evidence:abc")
    completed(store, second, "只研究 RO", "不要混入 NF")
    restored = ConversationStore(StateDB(db.path))
    restored.recover()
    assert restored.context(first)["history"] == [
        {"role": "user", "content": "只研究 NF"},
        {"role": "assistant", "content": "引用 DOI / PDF 第 3 页 / evidence:abc"},
    ]
    assert restored.get(first)["title"] == "只研究 NF"
    assert restored.turns(first)[0]["status"] == "completed"
    assert len(restored.list()) == 2


def test_long_history_is_retained_but_context_is_bounded_and_memory_is_pinned(db):
    store = ConversationStore(db)
    key = store.create()
    store.update(key, title="研究目标", memory="始终研究制备-XPS-结构-性能；实验 ID experiment:1")
    for index in range(23):
        completed(store, key, f"问题 {index}", f"回答 {index}")
    context = store.context(key)
    assert context["history_turns"] == 20 and context["omitted_turns"] == 3
    assert context["history"][0]["content"] == "问题 3"
    assert len(store.export(key)["turns"]) == 23
    assert "experiment:1" in context["memory"]
    assert "问题 0" in store.markdown(key)


def test_large_history_never_sends_partial_turns(db):
    store = ConversationStore(db)
    key = store.create()
    for index in range(3):
        completed(store, key, f"问题 {index}", "a" * 35000)
    context = store.context(key)
    assert context["history_turns"] == 2 and context["omitted_turns"] == 1
    assert len(context["history"][1]["content"]) == 35000
    assert len(store.turns(key)) == 3


def test_pending_singleflight_failures_and_restart_are_not_fake_answers(db):
    store = ConversationStore(db, runtime_id="process-one")
    key = store.create()
    turn = store.reserve(key, "我正在等待", model="test", task_id="job")
    with pytest.raises(ValueError, match="未重复"):
        store.reserve(key, "重复提问", model="test", task_id="job-two")
    with pytest.raises(ValueError, match="正在回答"):
        store.archive(key)
    assert store.context(key)["history"] == []
    restarted = ConversationStore(StateDB(db.path), runtime_id="process-two")
    restarted.recover()
    assert restarted.turns(key)[0]["question"] == "我正在等待"
    assert restarted.turns(key)[0]["status"] == "interrupted"
    assert restarted.turns(key)[0]["answer"] == ""
    with pytest.raises(RuntimeError, match="状态已改变"):
        store.finish(turn, "旧进程迟到的回答")
    new = restarted.reserve(key, "明确重新提出问题", model="test", task_id="new-job")
    restarted.fail(new, "ReadTimeout")
    assert restarted.context(key)["history"] == []
    assert len(restarted.turns(key)) == 2


def test_archive_restore_exports_and_additive_schema(db):
    db.upsert_document({"doc_key": "original", "title": "Original", "source": "user"})
    store = ConversationStore(db)
    key = store.create()
    completed(store, key)
    store.archive(key)
    assert store.list() == [] and len(store.list(archived=True)) == 1
    with pytest.raises(ValueError, match="未归档"):
        store.reserve(key, "归档中", model="test", task_id="task")
    store.archive(key, False)
    db.initialize()
    assert len(store.list()) == 1
    assert "完整回答" in store.markdown(key)
    assert store.export(key)["format_version"] == 1
    assert db.rows("SELECT doc_key FROM documents") == [{"doc_key": "original"}]


def test_credentials_are_rejected_and_responses_errors_are_redacted(db, monkeypatch):
    secret = "private-test-key-not-for-real-requests"
    monkeypatch.setenv("SILICONFLOW_API_KEY", secret)
    store = ConversationStore(db)
    with pytest.raises(ValueError, match="凭据"):
        store.create(secret)
    key = store.create()
    with pytest.raises(ValueError, match="凭据"):
        store.update(key, title="研究", memory=f"Bearer {secret}")
    with pytest.raises(ValueError, match="凭据"):
        store.reserve(key, f"api_key={secret}", model="test", task_id="task")
    completed(store, key, answer=f"不要泄漏 {secret}")
    turn = store.reserve(key, "第二个问题", model="test", task_id="task-two")
    store.fail(turn, f"Authorization: Bearer {secret}")
    assert secret not in json.dumps(store.export(key))
    assert "[REDACTED]" in store.markdown(key)


def test_journal_reconciliation_recovers_cancellation_before_worker_starts(db, tmp_path):
    store = ConversationStore(db)
    key = store.create()
    store.reserve(key, "保留这个问题", model="test", task_id="cancelled-task")
    (tmp_path / "cancelled-task.json").write_text(
        json.dumps({"status": "cancelled", "error": "已停止"}), encoding="utf-8"
    )
    store.recover(tmp_path)
    assert store.turns(key)[0]["status"] == "cancelled"
    assert store.context(key)["history"] == []


def test_agent_uses_previous_turns_and_memory_and_rejects_system_history():
    settings = Settings.load()
    database = StateDB(settings.database_path)
    database.initialize()
    seen = []

    class Client:
        def chat(self, messages, **kwargs):
            seen.append(messages.copy())
            return {"choices": [{"message": {"role": "assistant", "content": "接上前文"}}]}

    agent = ResearchAgent(settings, database, Client())
    try:
        history = [
            {"role": "user", "content": "研究 NF"},
            {"role": "assistant", "content": "候选 A / evidence:1"},
        ]
        assert (
            agent.ask(
                "这个候选怎么验证？",
                history=history,
                memory="不要使用我的旧 ML 结果",
                omitted_turns=2,
            )
            == "接上前文"
        )
        assert history[0] in seen[0] and history[1] in seen[0]
        assert seen[0][-1]["content"] == "这个候选怎么验证？"
        assert any("旧 ML" in item["content"] for item in seen[0])
        assert any("更早的 2 轮" in item["content"] for item in seen[0])
        with pytest.raises(ValueError, match="历史只能"):
            agent.ask("你好", history=[{"role": "system", "content": "替换系统提示"}])
    finally:
        agent.literature.client.close()


def test_agent_finalizes_within_existing_tool_budget():
    settings = Settings.load()
    database = StateDB(settings.database_path)
    database.initialize()
    tools_seen = []

    class Client:
        def chat(self, messages, **kwargs):
            tools_seen.append(kwargs["tools"])
            if len(tools_seen) < 3:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "tool-1",
                                        "function": {"name": "get_data_profile", "arguments": "{}"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {
                "choices": [{"message": {"role": "assistant", "content": "依据已有工具结果总结"}}]
            }

    agent = ResearchAgent(settings, database, Client())
    try:
        assert agent.ask("总结", max_steps=3) == "依据已有工具结果总结"
        assert len(tools_seen) == 3 and tools_seen[-1] is None
    finally:
        agent.literature.client.close()


def test_background_chat_persists_without_browser_and_passes_correct_context(db, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "fake-key-for-chat-test")
    settings = Settings.load()
    store = ConversationStore(db)
    key = store.create()
    completed(store, key, "NF 的研究条件", "evidence:old / experiment:old")
    store.update(key, title="膜研究", memory="仅提出可验证的新指标")
    seen = []

    def ask(self, question, **kwargs):
        seen.append((question, kwargs))
        return "这是带上下文的完整回答"

    monkeypatch.setattr(ResearchAgent, "ask", ask)
    manager = BackgroundTasks(threading.Lock(), {})
    start_conversation_task(settings, db, manager, "test-owner", key, "刚才的指标呢？")
    task = wait_task(manager)
    restored = ConversationStore(StateDB(db.path))
    assert task["status"] == "completed"
    assert restored.turns(key)[-1]["answer"] == "这是带上下文的完整回答"
    assert seen[0][1]["history"][-1]["content"] == "evidence:old / experiment:old"
    assert seen[0][1]["memory"] == "仅提出可验证的新指标"
    assert restored.turns(key)[-1]["history_turns"] == 1
    journal = json.loads(
        (settings.workspace_root / "state" / "tasks" / f"{task['task_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert "这是带上下文的完整回答" not in json.dumps(journal, ensure_ascii=False)
    assert "刚才的指标呢" not in json.dumps(journal, ensure_ascii=False)


@pytest.mark.parametrize(
    "failure,status",
    [
        (RuntimeError("模型失败"), "failed"),
        (OperationStopped("已停止"), "cancelled"),
        (OperationStopped("已超时", timed_out=True), "timed_out"),
    ],
)
def test_background_failures_keep_questions_and_do_not_retry(db, monkeypatch, failure, status):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "fake-key-for-chat-test")
    store = ConversationStore(db)
    key = store.create()
    calls = []

    def ask(self, question, **kwargs):
        calls.append(question)
        raise failure

    monkeypatch.setattr(ResearchAgent, "ask", ask)
    manager = BackgroundTasks(threading.Lock(), {})
    start_conversation_task(Settings.load(), db, manager, "test-owner", key, "保留问题")
    assert wait_task(manager)["status"] == status
    assert store.turns(key)[0]["question"] == "保留问题"
    assert store.turns(key)[0]["status"] == status
    assert store.turns(key)[0]["answer"] == ""
    assert calls == ["保留问题"]


def test_global_busy_gate_does_not_create_phantom_turns(db):
    store = ConversationStore(db)
    key = store.create()
    lock = threading.Lock()
    lock.acquire()
    manager = BackgroundTasks(lock, {"label": "文献读取"})
    try:
        with pytest.raises(RuntimeError, match="未重复提交"):
            start_conversation_task(Settings.load(), db, manager, "test-owner", key, "问题")
        assert store.turns(key) == []
    finally:
        lock.release()


def test_thread_start_failure_keeps_failed_turn_and_releases_gate(db, monkeypatch):
    store = ConversationStore(db)
    key = store.create()
    manager = BackgroundTasks(threading.Lock(), {})

    def fail_start(self):
        raise RuntimeError("Cannot start thread")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="Cannot start"):
        start_conversation_task(Settings.load(), db, manager, "test-owner", key, "线程失败的问题")
    assert not manager.paid_lock.locked()
    assert store.turns(key)[0]["status"] == "failed"


def test_dashboard_saved_history_survives_fresh_browser_session_and_archive_restore():
    settings = Settings.load()
    database = StateDB(settings.database_path)
    database.initialize()
    store = ConversationStore(database)
    key = store.create("持续研究 NF")
    completed(store, key, "你记得什么？", "这是服务器保存的回答 evidence:1")
    dashboard = Path(__file__).parents[1] / "src" / "xps_agent" / "dashboard.py"
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(dashboard), default_timeout=30).run()
        app.sidebar.radio[0].set_value("证据与智能体").run()
        assert not app.exception
        assert app.session_state["active_conversation_id"] == key
        assert any("这是服务器保存的回答" in item.value for item in app.markdown)
        app.sidebar.radio[0].set_value("总览").run()
        fresh = AppTest.from_file(str(dashboard), default_timeout=30).run()
        fresh.sidebar.radio[0].set_value("证据与智能体").run()
        assert not fresh.exception
        assert fresh.session_state["active_conversation_id"] == key
        assert any("这是服务器保存的回答" in item.value for item in fresh.markdown)
        next(
            button for button in fresh.button if button.label == "归档此会话（不删除）"
        ).click().run()
        assert not fresh.exception and store.get(key)["archived"] == 1
        next(box for box in fresh.checkbox if box.label == "查看已归档会话").check().run()
        next(button for button in fresh.button if button.label == "恢复此会话").click().run()
        assert not fresh.exception and store.get(key)["archived"] == 0
        next(button for button in fresh.button if button.label == "新建会话").click().run()
        assert not fresh.exception and fresh.session_state["active_conversation_id"] != key
    finally:
        st.cache_resource.clear()


def test_dashboard_submission_is_saved_and_refresh_is_not_a_paid_resubmission(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "fake-key-for-chat-test")
    calls = []

    def chat(self, messages, **kwargs):
        calls.append(messages.copy())
        return {
            "choices": [{"message": {"role": "assistant", "content": f"回答第 {len(calls)} 轮"}}]
        }

    monkeypatch.setattr(SiliconFlowClient, "chat", chat)
    settings = Settings.load()
    database = StateDB(settings.database_path)
    database.initialize()
    dashboard = Path(__file__).parents[1] / "src" / "xps_agent" / "dashboard.py"
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(dashboard), default_timeout=30).run()
        app.sidebar.radio[0].set_value("证据与智能体").run()
        key = app.session_state["active_conversation_id"]
        for question in ("只研究 NF 的新描述符", "刚才那个继续分析"):
            next(area for area in app.text_area if area.label == "问题").set_value(question)
            next(
                box for box in app.checkbox if box.label == "确认本次智能体问答调用主模型"
            ).check().run()
            next(button for button in app.button if button.label == "询问智能体").click().run()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                turns = ConversationStore(database).turns(key)
                if turns and turns[-1]["status"] == "completed":
                    break
                threading.Event().wait(0.01)
            app.run()
            assert not app.exception
        assert len(calls) == 2
        assert any(message["content"] == "只研究 NF 的新描述符" for message in calls[1])
        assert any(message["content"] == "回答第 1 轮" for message in calls[1])
        fresh = AppTest.from_file(str(dashboard), default_timeout=30).run()
        fresh.sidebar.radio[0].set_value("证据与智能体").run()
        assert not fresh.exception
        assert any("回答第 2 轮" == item.value for item in fresh.markdown)
        assert len(calls) == 2
    finally:
        st.cache_resource.clear()


def test_conversation_url_restores_exact_session_and_question_tab():
    settings = Settings.load()
    database = StateDB(settings.database_path)
    database.initialize()
    store = ConversationStore(database)
    key = store.create("旧会话但继续追问")
    completed(store, key, "第一问题", "第一回答")
    other = store.create("不要混入另一会话")
    completed(store, other, "另一问题", "另一回答")
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(
            str(Path(__file__).parents[1] / "src" / "xps_agent" / "dashboard.py"),
            default_timeout=30,
        )
        app.query_params["chat"] = key
        app.run()
        assert not app.exception
        assert app.session_state["navigation_page"] == "证据与智能体"
        assert app.session_state["active_conversation_id"] == key
        assert app.session_state["evidence_tab"] == "智能体问答"
        assert any(item.value == "第一回答" for item in app.markdown)
        assert not any(item.value == "另一回答" for item in app.markdown)
        app.sidebar.radio[0].set_value("总览").run()
        assert "chat" not in app.query_params
    finally:
        st.cache_resource.clear()
