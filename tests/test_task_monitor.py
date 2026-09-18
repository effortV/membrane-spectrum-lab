import json
import threading
import time

import streamlit as st
from streamlit.testing.v1 import AppTest
from pathlib import Path

from xps_agent.task_monitor import (
    progress_measure,
    public_progress,
    read_task_history,
    summarize_result,
)
from xps_agent.tasks import BackgroundTasks


def _wait(manager, owner="owner"):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = manager.snapshot(owner)
        if result and result["status"] != "running":
            return result
        threading.Event().wait(0.01)
    raise AssertionError("task did not finish")


def test_cross_session_monitor_shows_live_progress_but_not_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sensitive-monitor-key")
    manager = BackgroundTasks(threading.Lock(), {})
    entered, release = threading.Event(), threading.Event()

    def worker(context):
        context.update(
            stage="等待模型响应",
            document_title="paper sensitive-monitor-key",
            page=3,
            documents_total=5,
            documents_finished=2,
            question="private question",
            model="test",
            wait_limit_seconds=90,
        )
        entered.set()
        release.wait(3)
        return {"documents": 5, "pages_completed": 12, "answer": "private answer"}

    task_id = manager.start("owner", "读文献", "extraction", worker, journal_root=tmp_path)
    try:
        assert entered.wait(1)
        records = read_task_history(tmp_path, manager.snapshot("another"))
        task = records[0]
        assert task["task_id"] == task_id and task["status"] == "running"
        assert task["progress"]["page"] == 3
        assert task["progress"]["documents_finished"] == 2
        assert not task["is_owner"]
        serialized = json.dumps(records)
        assert "private question" not in serialized and "sensitive-monitor-key" not in serialized
    finally:
        release.set()
        _wait(manager)
    history = read_task_history(tmp_path, manager.snapshot("another"))
    assert history[0]["status"] == "completed"
    assert history[0]["result_summary"] == {"documents": 5, "pages_completed": 12}
    assert "private answer" not in json.dumps(history)
    assert len(history[0]["events"]) >= 3
    assert not manager.paid_lock.locked()


def test_history_survives_restart_and_does_not_claim_stale_job_running(tmp_path):
    manager = BackgroundTasks(threading.Lock(), {})
    manager.start(
        "owner",
        "检索",
        "search",
        lambda context: {"added_or_updated": 30, "answer": "private"},
        journal_root=tmp_path,
    )
    _wait(manager)
    (tmp_path / "stale.json").write_text(
        json.dumps(
            {
                "task_id": "stale",
                "label": "旧任务",
                "kind": "extraction",
                "status": "running",
                "created_at": "2026-09-17T08:00:00+00:00",
                "progress": {"page": 2},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "broken.json").write_text("{broken", encoding="utf-8")
    fresh = BackgroundTasks(threading.Lock(), {})
    records = read_task_history(tmp_path, fresh.snapshot("new"))
    assert len(records) == 2
    assert next(item for item in records if item["task_id"] == "stale")["status"] == "interrupted"
    finished = next(item for item in records if item["kind"] == "search")
    assert finished["result_summary"] == {"added_or_updated": 30}
    assert "private" not in json.dumps(records)


def test_progress_uses_actual_counts_not_a_fake_timer():
    assert progress_measure({"stage": "等待模型响应", "wait_limit_seconds": 90}) is None
    assert progress_measure({"documents_total": 0}) is None
    assert progress_measure({"fits_total": 30, "fits_finished": 12}) == (0.4, "训练／验证 12/30")
    assert progress_measure({"documents_total": 5, "documents_finished": 9}) == (1, "文献 5/5")
    assert public_progress(
        {"stage": "running", "prompt": "private", "payload": {}, "page": float("nan")}
    ) == {"stage": "running"}
    assert summarize_result(
        {"index": {"indexed": 2}, "text": {"parsed": 2}, "answer": "private"}
    ) == {"index.indexed": 2, "text.parsed": 2}


def test_journal_large_progress_stays_valid_json_and_has_bounded_timeline(tmp_path):
    manager = BackgroundTasks(threading.Lock(), {})

    def worker(context):
        for page in range(80):
            context.update(
                stage="read", page=page, document_title="文献" * 10000, payload="private"
            )
        return {"pages_completed": 80}

    task_id = manager.start("owner", "test", "extraction", worker, journal_root=tmp_path)
    _wait(manager)
    journal = json.loads((tmp_path / f"{task_id}.json").read_text(encoding="utf-8"))
    assert len(journal["events"]) <= 40
    assert len(journal["progress"]["document_title"]) <= 1200
    assert "private" not in json.dumps(journal)
    assert journal["result_summary"]["pages_completed"] == 80


def test_monitor_page_read_only_history_and_result_navigation(tmp_path, monkeypatch):
    monkeypatch.setenv("XPS_PROJECT_ROOT", str(tmp_path))
    root = tmp_path / "workspace" / "state" / "tasks"
    root.mkdir(parents=True)
    (root / "historical.json").write_text(
        json.dumps(
            {
                "task_id": "historical",
                "label": "检索 OpenAlex",
                "kind": "search",
                "status": "completed",
                "created_at": "2026-09-17T08:00:00+00:00",
                "finished_at": "2026-09-17T08:00:06+00:00",
                "progress": {"stage": "检索 OpenAlex"},
                "result_summary": {"added_or_updated": 30},
            }
        ),
        encoding="utf-8",
    )
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(
            str(Path(__file__).parents[1] / "src" / "xps_agent" / "dashboard.py"), default_timeout=5
        ).run()
        next(button for button in app.sidebar.button if button.label == "后台任务").click().run()
        assert not app.exception
        assert any(item.label == "正在运行" and item.value == "0" for item in app.metric)
        assert len(app.dataframe) == 1
        assert any("当前没有正在运行" in item.value for item in app.success)
        next(
            button for button in app.button if button.label == "前往结果页面：文献中心"
        ).click().run()
        assert not app.exception
        assert app.sidebar.radio[0].value == "文献中心"
        assert not (tmp_path / "workspace" / "state" / "xps_agent.sqlite3").stat().st_size == 0
        assert len(list(root.glob("*.json"))) == 1  # Viewing did not submit a task.
    finally:
        st.cache_resource.clear()
