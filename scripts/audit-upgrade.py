"""Verify a new implementation against live inputs in an isolated workspace, without APIs."""

from dataclasses import replace
from pathlib import Path
import json
import shutil

from xps_agent.config import Settings
from xps_agent.data import DataAuditor
from xps_agent.db import StateDB
from xps_agent.diagnostics import health_report
from xps_agent.hypotheses import HypothesisEngine
from xps_agent.utils import json_dumps


test_root = Path(__file__).resolve().parents[1]
live_root = Path(r"D:\zzh\XPS-agent")
if not test_root.is_relative_to(live_root / "workspace" / "upgrade_staging"):
    raise SystemExit(
        "This verification script is restricted to the isolated upgrade_staging directory."
    )
backups = sorted((live_root / "workspace" / "backups").glob("upgrade_*"))
backup = backups[-1]
Settings.load(live_root)  # Load credentials for redaction only; never initialize an API client.
settings = Settings.load(test_root)
settings.ensure_workspace()
shutil.copy2(backup / "xps_agent.sqlite3", settings.database_path)
db = StateDB(settings.database_path)
db.initialize()
settings = replace(
    settings,
    data_root=Path(r"D:\data\XPS-914"),
    reference_root=Path(r"D:\data\XPS-914\reference"),
    legacy_root=live_root / "final",
)
previous = json.loads((backup / "canonical" / "data_profile.json").read_text(encoding="utf-8"))
before = health_report(settings, db)
profile = DataAuditor(settings, db).run()
materialized = HypothesisEngine(settings, db).materialize()
report = {
    "previous_rows": previous.get("model_table_rows"),
    "new_rows": profile.get("model_table_rows"),
    "previous_usable_spectra": previous.get("canonical_spectra_usable"),
    "new_usable_spectra": profile.get("canonical_spectra_usable"),
    "previous_doi_rows": previous.get("doi_group_rows_resolved"),
    "new_doi_rows": profile.get("doi_group_rows_resolved"),
    "materialization": materialized,
    "health": health_report(settings, db),
    "preparation_failures_before": before["failed_documents"],
}
destination = settings.workspace_root / "upgrade_audit.json"
destination.write_text(json_dumps(report), encoding="utf-8")
print(json_dumps(report))
