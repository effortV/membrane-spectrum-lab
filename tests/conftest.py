import pytest
from pathlib import Path
import shutil
from xps_agent.config import MODEL_SETTING_NAMES


@pytest.fixture(autouse=True)
def isolate_central_storage(monkeypatch, tmp_path, request):
    for name in MODEL_SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    # Never load production .env or mutate active/archived scientific state in tests.
    root = tmp_path / "default_project"
    root.mkdir()
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / "prompts", root / "prompts")
    canonical = source / "workspace" / "canonical"
    if request.node.name == "test_dashboard_overview_renders" and canonical.exists():
        shutil.copytree(canonical, root / "workspace" / "canonical")
    for name, value in {
        "XPS_PROJECT_ROOT": str(root),
        "XPS_STORAGE_ROOT": "",
        "XPS_WORKSPACE_ROOT": "",
        "XPS_DATA_ROOT": str(root / "raw" / "tables"),
        "XPS_REFERENCE_ROOT": str(root / "raw" / "literature"),
        "XPS_LEGACY_ROOT": str(root / "raw" / "spectra"),
    }.items():
        monkeypatch.setenv(name, value)
