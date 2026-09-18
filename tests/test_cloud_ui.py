from pathlib import Path
import sys

import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import backend_client  # noqa: E402


def fake_config():
    return backend_client.ConnectionSettings.from_mapping(
        {
            "XPS_SSH_HOST": "example.invalid",
            "XPS_SSH_PRIVATE_KEY": "fake",
            "XPS_SSH_HOST_KEY_SHA256": "SHA256:" + "A" * 43,
            "XPS_BACKEND_TOKEN": "x" * 48,
        }
    )


def test_thin_cloud_ui_all_pages_no_local_database(monkeypatch):
    monkeypatch.setattr(backend_client.ConnectionSettings, "from_mapping", lambda _: fake)
    fake = backend_client.ConnectionSettings(
        "example.invalid", 2222, "xpscloud", "fake", "SHA256:" + "A" * 43, "x" * 48
    )

    def response(self, method, path, owner, **kwargs):
        if path == "/v1/overview":
            return {
                "counts": {"documents": 2, "evidence": 3, "hypotheses": 0, "experiments": 0},
                "profile": {},
                "settings": {
                    "main_model": "test-main",
                    "vision_model": "test-vision",
                    "siliconflow_key": "configured",
                },
            }
        if path == "/v1/tasks":
            return {"current": None, "history": []}
        if path == "/v1/data":
            return {"total": 0, "columns": ["record_id", "dataset"], "rows": []}
        if path == "/v1/research/catalog":
            return {
                "columns": [],
                "relationships": {
                    "xps_to_structure": {
                        "label": "XPS → 结构",
                        "inputs": ["xps"],
                        "target": "structure",
                    }
                },
                "roles": {"xps": "XPS"},
                "datasets": ["NF"],
            }
        return []

    monkeypatch.setattr(backend_client.SSHBackendClient, "request", response)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(ROOT / "streamlit_cloud.py"), default_timeout=30).run()
        assert not app.exception
        for page in (
            "数据与谱图",
            "文献中心",
            "证据与智能体",
            "XPS 图分析",
            "关系研究（ML）",
            "系统设置",
        ):
            app.sidebar.radio[0].set_value(page).run()
            assert not app.exception, page
        assert "后台任务" not in app.sidebar.radio[0].options
        next(button for button in app.sidebar.button if button.label == "后台任务").click().run()
        assert not app.exception
        assert any(title.value == "现在正在做什么" for title in app.title)
        from xps_agent.config import Settings

        assert not Settings.load().database_path.exists()
    finally:
        st.cache_resource.clear()


def test_thin_cloud_connection_failure_stops_cleanly(monkeypatch):
    def fail(_):
        raise backend_client.BackendError("connection_not_configured")

    monkeypatch.setattr(backend_client.ConnectionSettings, "from_mapping", fail)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(ROOT / "streamlit_cloud.py"), default_timeout=30).run()
        assert not app.exception and not app.sidebar.radio
        assert any("connection_not_configured" in item.value for item in app.error)
        from xps_agent.config import Settings

        assert not Settings.load().database_path.exists()
    finally:
        st.cache_resource.clear()
