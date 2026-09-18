from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from xps_agent.config import Settings
from xps_agent.db import StateDB
from xps_agent.library_activity import LibraryActivity
from xps_agent.library_ui import document_table
from xps_agent.literature import LiteratureError, LiteratureService
from xps_agent.tasks import OperationStopped


def _service(tmp_path, *, works=None, status=200, check=None):
    settings = replace(Settings.load(), download_rate_seconds=0)
    database = StateDB(tmp_path / "library.sqlite3")
    database.initialize()
    service = LiteratureService(settings, database, check_cancelled=check)
    service.client.close()
    service.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json={"results": works or []})
        )
    )
    return service, database


def _work(name):
    return {
        "doi": f"https://doi.org/10.1000/{name}",
        "title": f"膜研究 {name}",
        "publication_year": 2025,
        "id": f"https://openalex.org/{name}",
        "best_oa_location": {
            "pdf_url": f"https://example.org/{name}.pdf",
            "landing_page_url": f"https://example.org/{name}",
        },
    }


def test_search_receipt_distinguishes_new_existing_and_survives_reopen(tmp_path):
    service, db = _service(tmp_path, works=[_work("old"), _work("new"), _work("new")])
    db.upsert_document(
        {
            "doc_key": "doi:10.1000/old",
            "title": "旧文献",
            "source": "local",
            "status": "parsed",
            "local_path": "old.pdf",
        }
    )
    try:
        keys = service.search_openalex("polyamide XPS", 30)
        first = service.last_batch_id
        assert len(keys) == 2
        assert service.activity.summary(first) == {
            "batch_id": first,
            "matched": 2,
            "added": 1,
            "existing": 1,
        }
        assert (
            db.rows("SELECT status FROM documents WHERE doc_key='doi:10.1000/old'")[0]["status"]
            == "parsed"
        )
        assert len(db.rows("SELECT * FROM download_queue")) == 1
        service.search_openalex("membrane spectroscopy", 30)
        assert service.activity.summary(service.last_batch_id)["added"] == 0
        reopened = LibraryActivity(StateDB(db.path))
        assert len(reopened.batches()) == 2
        assert reopened.summary(first)["added"] == 1
        assert reopened.batches()[1]["query"] == "polyamide XPS"
    finally:
        service.client.close()


def test_failed_search_is_visible_and_empty_results_are_not_fake_success(tmp_path):
    service, db = _service(tmp_path, status=403)
    try:
        with pytest.raises(LiteratureError):
            service.search_openalex("XPS", 10)
        batch = service.activity.batches()[0]
        assert batch["status"] == "failed" and batch["total"] == 0
        assert service.activity.items(batch["batch_id"]) == []
        assert db.rows("SELECT * FROM documents") == []
    finally:
        service.client.close()


def test_empty_valid_search_has_zero_matching_new_and_existing(tmp_path):
    service, _ = _service(tmp_path)
    try:
        assert service.search_openalex("XPS", 10) == []
        assert service.activity.summary(service.last_batch_id)["matched"] == 0
        assert service.activity.batches()[0]["status"] == "completed"
    finally:
        service.client.close()


def test_download_scope_and_per_document_receipt(tmp_path, monkeypatch):
    service, db = _service(tmp_path, works=[_work("good"), _work("manual"), _work("outside")])
    try:
        service.search_openalex("XPS", 30)

        def download(url, destination, headers=None):
            if "manual" in url:
                raise LiteratureError("Full text not accessible")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"%PDF-test-offline")
            return destination

        monkeypatch.setattr(service, "_try_url", download)
        result = service.fetch_queued(30, ["doi:10.1000/good", "doi:10.1000/manual"])
        assert result["downloaded"] == 1 and result["manual"] == 1 and result["failed"] == 0
        rows = service.activity.items(result["batch_id"])
        assert {row["outcome"] for row in rows} == {"downloaded", "manual"}
        assert all(row["doc_key"] != "doi:10.1000/outside" for row in rows)
        assert next(row for row in rows if row["outcome"] == "downloaded")["local_path"]
        assert (
            db.rows("SELECT status FROM download_queue WHERE doc_key='doi:10.1000/outside'")[0][
                "status"
            ]
            == "queued"
        )
        assert "需要补传 PDF" in document_table(rows)["全文 / 读取状态"].tolist()
        assert service.activity.batches("fetch")[0]["total"] == 2
    finally:
        service.client.close()


def test_empty_download_scope_never_fetches_the_entire_queue(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, works=[_work("one")])
    try:
        service.search_openalex("XPS", 10)
        monkeypatch.setattr(
            service, "_try_url", lambda *args, **kwargs: pytest.fail("Unexpected network call")
        )
        result = service.fetch_queued(10, [])
        assert result["downloaded"] == 0
        assert service.activity.items(result["batch_id"]) == []
    finally:
        service.client.close()


def test_cancelled_download_keeps_successes_and_pending_items(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, works=[_work("one"), _work("two")])
    try:
        service.search_openalex("XPS", 10)
        count = 0

        def download(url, destination, headers=None):
            nonlocal count
            count += 1
            if count == 2:
                raise OperationStopped("Test cancellation")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"%PDF-test-offline")
            return destination

        monkeypatch.setattr(service, "_try_url", download)
        with pytest.raises(OperationStopped):
            service.fetch_queued(10)
        batch = service.activity.batches("fetch")[0]
        assert batch["status"] == "cancelled"
        assert {row["outcome"] for row in service.activity.items(batch["batch_id"])} == {
            "pending",
            "downloaded",
        }
    finally:
        service.client.close()


def test_legacy_search_is_labelled_without_inventing_new_counts(tmp_path):
    db = StateDB(tmp_path / "legacy.sqlite3")
    db.initialize()
    db.upsert_document(
        {
            "doc_key": "legacy",
            "title": "已有联网文献",
            "source": "openalex",
            "metadata": {"search_query": "XPS"},
        }
    )
    rows = LibraryActivity(db).legacy_search_items()
    assert len(rows) == 1 and rows[0]["is_new"] is None
    assert document_table(rows, receipts=True)["本次入库"].tolist() == ["无法追溯"]
    assert document_table(rows, receipts=True).columns.tolist()[:3] == [
        "本次入库",
        "文献",
        "全文 / 读取状态",
    ]
    assert LibraryActivity(db).batches() == []


def test_schema_upgrade_is_additive_and_preserves_existing_document(tmp_path):
    db = StateDB(tmp_path / "migration.sqlite3")
    db.initialize()
    db.upsert_document(
        {
            "doc_key": "original",
            "title": "不可更改的原始文献",
            "source": "local",
            "status": "parsed",
            "local_path": "original.pdf",
        }
    )
    before = db.rows("SELECT * FROM documents")
    with db.connect() as con:
        con.execute("DROP TABLE literature_batch_items")
        con.execute("DROP TABLE literature_batches")
    db.initialize()
    assert db.rows("SELECT * FROM documents") == before
    assert LibraryActivity(db).batches() == []


def test_literature_page_shows_receipts_and_reading_distinction_without_api_calls():
    settings = Settings.load()
    db = StateDB(settings.database_path)
    db.initialize()
    db.upsert_document(
        {
            "doc_key": "visible",
            "title": "应显示的新增文献",
            "source": "openalex",
            "doi": "10.1000/visible",
            "status": "queued",
        }
    )
    activity = LibraryActivity(db)
    key = activity.start("search", "XPS membrane")
    activity.item(key, "visible", 1, is_new=True, outcome="new")
    activity.finish(key)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(
            str(Path(__file__).parents[1] / "src/xps_agent/dashboard.py"), default_timeout=20
        ).run()
        app.sidebar.radio[0].set_value("文献中心").run()
        assert not app.exception
        assert any("命中 1 篇 · 新增 1 篇 · 已有 0 篇" in value.value for value in app.markdown)
        assert any("应显示的新增文献" in frame.value.to_string() for frame in app.dataframe)
        assert any("这一步只做计划" in value.value for value in app.markdown)
        assert any(button.label == "读取待处理 PDF 正文" for button in app.button)
        assert any(button.label == "更新 XPS 图页清单" for button in app.button)
        assert "后台任务" not in app.sidebar.radio[0].options
        assert any(button.label == "后台任务" for button in app.sidebar.button)
        assert db.rows("SELECT COUNT(*) AS n FROM llm_request_events")[0]["n"] == 0
    finally:
        st.cache_resource.clear()
