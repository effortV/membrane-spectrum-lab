import os
import shutil
import threading
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest
import streamlit as st

from xps_agent.config import save_api_settings
from xps_agent.db import StateDB
from xps_agent.pdfs import PDFService
from xps_agent.llm import SiliconFlowClient
from xps_agent.security import hash_ui_password


def test_dashboard_overview_renders():
    dashboard = Path(__file__).parents[1] / "src" / "xps_agent" / "dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30)
    app.run()
    assert not app.exception
    assert app.sidebar.radio

    app.sidebar.radio[0].set_value("系统设置").run()
    assert not app.exception
    assert any(button.label == "保存 API 配置" for button in app.button)
    assert len(app.text_input) == 5
    assert not any("界面访问保护" in item.value for item in app.subheader)
    for page in [
        "数据与谱图",
        "文献中心",
        "证据与智能体",
        "关系研究（ML）",
        "XPS 图分析",
    ]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
    assert "后台任务" not in app.sidebar.radio[0].options
    next(button for button in app.sidebar.button if button.label == "后台任务").click().run()
    assert not app.exception
    assert any(item.value == "现在正在做什么" for item in app.subheader)
    app.sidebar.radio[0].set_value("文献中心").run()
    assert not app.exception
    assert not app.session_state["show_task_monitor"]


def test_dashboard_no_longer_has_password_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("XPS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("XPS_UI_PASSWORD_HASH", hash_ui_password("test-password-123"))
    st.cache_resource.clear()
    try:
        dashboard = Path(__file__).parents[1] / "src" / "xps_agent" / "dashboard.py"
        app = AppTest.from_file(str(dashboard), default_timeout=30).run()
        assert not app.exception
        assert app.sidebar.radio
        assert not any(button.label == "登录" for button in app.button)
    finally:
        st.cache_resource.clear()


def test_visual_job_does_not_block_page_navigation(tmp_path, monkeypatch):
    monkeypatch.setenv("XPS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("SILICONFLOW_API_KEY", "fake-key-for-ui-test")
    monkeypatch.setenv("XPS_UI_PASSWORD_HASH", "")
    shutil.copytree(Path(__file__).parents[1] / "prompts", tmp_path / "prompts")
    db = StateDB(tmp_path / "workspace" / "state" / "xps_agent.sqlite3")
    db.initialize()
    db.upsert_document(
        {
            "doc_key": "test",
            "title": "test",
            "source": "test",
            "local_path": "fake.pdf",
            "status": "parsed",
        }
    )
    monkeypatch.setattr(
        PDFService,
        "parse_pdf",
        lambda self, path: {
            "sha256": "fake-pdf",
            "pages": [{"page": 1, "text": "XPS", "relevance_score": 5}],
        },
    )
    monkeypatch.setattr(
        PDFService, "render_pages", lambda self, path, pages, digest: [Path("fake.png")]
    )
    entered, release = threading.Event(), threading.Event()

    def vision(self, prompt, images, **kwargs):
        entered.set()
        release.wait(5)
        return {"items": [], "figures": []}

    monkeypatch.setattr(SiliconFlowClient, "vision_json", vision)
    st.cache_resource.clear()
    try:
        dashboard = Path(__file__).parents[1] / "src" / "xps_agent" / "dashboard.py"
        app = AppTest.from_file(str(dashboard), default_timeout=3).run()
        app.sidebar.radio[0].set_value("证据与智能体").run()
        next(box for box in app.checkbox if box.label == "确认调用付费视觉模型").check().run()
        next(button for button in app.button if button.label == "开始页级证据抽取").click().run()
        assert entered.wait(1)
        assert not app.exception
        app.sidebar.radio[0].set_value("总览").run()
        assert not app.exception
        assert any(button.label.startswith("停止后续请求") for button in app.button)
        release.set()
        deadline = time.monotonic() + 2
        while (
            time.monotonic() < deadline
            and db.rows("SELECT status FROM documents")[0]["status"] != "evidence_extracted"
        ):
            threading.Event().wait(0.01)
        app.sidebar.radio[0].set_value("证据与智能体").run()
        assert not app.exception
        assert app.session_state["last_extraction_result"]["documents"] == 1
    finally:
        release.set()
        st.cache_resource.clear()


def test_hot_deploy_of_new_dashboard_blocks_old_runtime(monkeypatch):
    monkeypatch.setattr(SiliconFlowClient, "__init__", lambda self, settings, db=None: None)
    st.cache_resource.clear()
    try:
        dashboard = Path(__file__).parents[1] / "src" / "xps_agent" / "dashboard.py"
        app = AppTest.from_file(str(dashboard)).run()
        assert not app.exception and not app.sidebar.radio
        assert any("当前进程仍加载旧模块" in warning.value for warning in app.warning)
    finally:
        st.cache_resource.clear()


def test_local_pdf_reading_has_nonblocking_visible_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("XPS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("XPS_UI_PASSWORD_HASH", "")
    db = StateDB(tmp_path / "workspace" / "state" / "xps_agent.sqlite3")
    db.initialize()
    db.upsert_document(
        {
            "doc_key": "pending",
            "title": "test",
            "source": "test",
            "local_path": "pending.pdf",
            "status": "indexed",
        }
    )
    entered, release = threading.Event(), threading.Event()

    def parse(self, path):
        entered.set()
        release.wait(5)
        return {"pages": [], "page_count": 1}

    monkeypatch.setattr(PDFService, "parse_pdf", parse)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(
            str(Path(__file__).parents[1] / "src" / "xps_agent" / "dashboard.py"), default_timeout=3
        ).run()
        app.sidebar.radio[0].set_value("文献中心").run()
        next(button for button in app.button if button.label == "读取待处理 PDF 正文").click().run()
        assert entered.wait(1) and not app.exception
        app.sidebar.radio[0].set_value("总览").run()
        assert not app.exception
        assert any("任务状态" in item.value for item in app.subheader)
        next(button for button in app.sidebar.button if button.label == "后台任务").click().run()
        assert not app.exception
        assert any(item.value == "现在正在做什么" for item in app.subheader)
        assert any(item.label == "正在运行" and item.value == "1" for item in app.metric)
        release.set()
        deadline = time.monotonic() + 2
        while (
            time.monotonic() < deadline
            and db.rows("SELECT status FROM documents")[0]["status"] != "parsed"
        ):
            threading.Event().wait(0.01)
        app.sidebar.radio[0].set_value("文献中心").run()
        assert not app.exception
        assert app.session_state["last_literature_result"]["parsed"] == 1
    finally:
        release.set()
        st.cache_resource.clear()


def test_save_api_settings_updates_and_clears_without_exposing_values(tmp_path):
    (tmp_path / ".env.example").write_text(
        "SILICONFLOW_API_KEY=\nOPENALEX_API_KEY=\nXPS_MAIN_MODEL=test-model\n",
        encoding="utf-8",
    )
    secret = "test-secret-with-#-and-space"
    try:
        changed = save_api_settings(
            tmp_path,
            {"SILICONFLOW_API_KEY": secret, "OPENALEX_API_KEY": "openalex-test"},
        )
        contents = (tmp_path / ".env").read_text(encoding="utf-8")
        assert set(changed) == {"SILICONFLOW_API_KEY", "OPENALEX_API_KEY"}
        assert f'SILICONFLOW_API_KEY="{secret}"' in contents
        assert "XPS_MAIN_MODEL=test-model" in contents
        assert os.environ["SILICONFLOW_API_KEY"] == secret

        save_api_settings(tmp_path, {"SILICONFLOW_API_KEY": ""})
        assert "SILICONFLOW_API_KEY=" in (tmp_path / ".env").read_text(encoding="utf-8")
        assert "SILICONFLOW_API_KEY" not in os.environ
    finally:
        os.environ.pop("SILICONFLOW_API_KEY", None)
        os.environ.pop("OPENALEX_API_KEY", None)
