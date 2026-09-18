"""Create a recoverable pre-upgrade snapshot; never copy API credentials or raw data."""

from datetime import UTC, datetime
from pathlib import Path
import shutil
import sqlite3
from xps_agent.config import Settings


root = Path(__file__).resolve().parents[1]
stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
workspace = Settings.load(root).workspace_root
destination = workspace / "backups" / f"upgrade_{stamp}"
destination.mkdir(parents=True)
for name in ("src", "scripts", "prompts", "configs", "tests", ".streamlit"):
    if (root / name).exists():
        shutil.copytree(
            root / name, destination / name, ignore=shutil.ignore_patterns("__pycache__")
        )
for name in (
    "README.md",
    "ARCHITECTURE.md",
    "pyproject.toml",
    "uv.lock",
    "streamlit_app.py",
    ".env.example",
):
    if (root / name).exists():
        shutil.copy2(root / name, destination / name)
canonical = workspace / "canonical"
if canonical.exists():
    shutil.copytree(canonical, destination / "canonical")
database = workspace / "state" / "xps_agent.sqlite3"
if database.exists():
    with (
        sqlite3.connect(database) as source,
        sqlite3.connect(destination / database.name) as target,
    ):
        source.backup(target)
print(f"Backup created: {destination}")
