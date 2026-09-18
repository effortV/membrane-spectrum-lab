from __future__ import annotations

import json
from typing import Any

import pandas as pd

from . import __version__
from .config import Settings
from .db import StateDB
from .features import DescriptorSpec, SPECTRUM_SUMMARY_VERSION
from .hypotheses import validate_candidate
from .security import safe_error
from .utils import sha256_file, utc_now


def health_report(settings: Settings, db: StateDB) -> dict[str, Any]:
    """Offline, read-only scientific/runtime checks. No LLM or literature requests."""
    issues: list[dict[str, str]] = []

    def issue(severity: str, code: str, message: str) -> None:
        issues.append({"severity": severity, "code": code, "message": message})

    integrity = db.rows("PRAGMA quick_check")
    if any(next(iter(row.values())) != "ok" for row in integrity):
        issue(
            "error",
            "database_integrity",
            "SQLite quick_check failed; restore/check backup before writes.",
        )
    canonical = settings.workspace_root / "canonical"
    profile_path = canonical / "data_profile.json"
    table_path = canonical / "model_table.csv"
    inventory_path = canonical / "column_inventory.csv"
    rows = 0
    unresolved = 0
    roles: dict[str, str] = {}
    columns: set[str] = set()
    if profile_path.exists():
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if profile.get("spectrum_summary_version") != SPECTRUM_SUMMARY_VERSION:
                issue(
                    "warning",
                    "spectrum_version",
                    "Re-audit data to use energy-grid-weighted, versioned spectrum summaries.",
                )
            for sheet in profile.get("workbook_sheets", []):
                from pathlib import Path

                source = Path(sheet["source"])
                if not source.exists() or sheet.get("sha256") != sha256_file(source):
                    issue(
                        "warning",
                        "source_changed",
                        f"Source workbook missing/changed: {source.name}",
                    )
        except (ValueError, KeyError, OSError) as exc:
            issue("error", "profile_invalid", safe_error(exc))
    else:
        issue("error", "data_not_prepared", "Data profile is missing; run offline data audit.")
    if table_path.exists():
        try:
            frame = pd.read_csv(table_path, low_memory=False)
            columns = set(frame.columns)
            rows = len(frame)
            if "record_id" in frame and frame["record_id"].duplicated().any():
                issue(
                    "error",
                    "duplicate_records",
                    "Canonical table contains duplicate record_id values.",
                )
            if "doi_group" in frame:
                unresolved = int(frame["doi_group"].fillna("").astype(str).eq("").sum())
                if unresolved:
                    issue(
                        "warning",
                        "unresolved_doi",
                        f"{unresolved}/{rows} rows lack resolved DOI; strict ML excludes them by default.",
                    )
            issue(
                "warning",
                "sample_linkage",
                "Within-paper row-order spectrum/sample links are provisional, not verified chemical identity links.",
            )
        except (ValueError, OSError) as exc:
            issue("error", "table_invalid", safe_error(exc))
    if inventory_path.exists():
        inventory = pd.read_csv(inventory_path)
        roles = dict(
            zip(inventory["column"].astype(str), inventory["role"].astype(str), strict=True)
        )
    evidence_ids = {row["evidence_id"] for row in db.rows("SELECT evidence_id FROM evidence")}
    malformed_confidence = db.rows(
        "SELECT COUNT(*) count FROM evidence WHERE confidence<0 OR confidence>1"
    )[0]["count"]
    if malformed_confidence:
        issue(
            "warning",
            "legacy_confidence",
            f"{malformed_confidence} historical confidence values are outside [0,1]; review them without treating model self-scores as validation.",
        )
    candidate_checks = []
    symbols: set[str] = set(columns)
    for row in db.rows("SELECT hypothesis_id,status,spec_json FROM hypotheses"):
        check = {"hypothesis_id": row["hypothesis_id"], "status": row["status"]}
        try:
            spec = DescriptorSpec.model_validate(json.loads(row["spec_json"]))
            validate_candidate(spec, roles, evidence_ids, symbols)
            symbols.add(spec.symbol)
            check["validation"] = (
                "schema_inputs_and_citations_pass_not_novelty_or_mechanism_validation"
            )
        except (ValueError, TypeError, KeyError) as exc:
            check["validation"] = "needs_review"
            check["reason"] = safe_error(exc)
            issue(
                "warning",
                "candidate_needs_review",
                f"Candidate {row['hypothesis_id']} needs schema/input/citation review.",
            )
        candidate_checks.append(check)
    enriched = canonical / "model_table_enriched.csv"
    manifest_path = enriched.with_suffix(".manifest.json")
    if enriched.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not table_path.exists()
                or manifest.get("source_sha256") != sha256_file(table_path)
                or manifest.get("destination_sha256") != sha256_file(enriched)
                or manifest.get("spectrum_summary_version") != SPECTRUM_SUMMARY_VERSION
            ):
                issue(
                    "warning",
                    "enriched_stale",
                    "Enriched table is stale/changed; re-materialize before ML.",
                )
        except (ValueError, OSError) as exc:
            issue("warning", "enriched_manifest", safe_error(exc))
    failed_docs = [
        {
            "doc_key": row["doc_key"],
            "status": row["status"],
            "reason": safe_error(row["reason"] or ""),
        }
        for row in db.rows(
            "SELECT doc_key,status,reason FROM documents WHERE status IN ('evidence_failed','parse_failed','evidence_no_relevant_pages') ORDER BY updated_at DESC LIMIT 20"
        )
    ]
    if failed_docs:
        issue(
            "warning",
            "document_failures",
            f"{len(failed_docs)} failed/skipped documents shown; review before retrying paid extraction.",
        )
    return {
        "checked_at": utc_now(),
        "agent_version": __version__,
        "offline_only": True,
        "ready": not any(item["severity"] == "error" for item in issues),
        "rows": rows,
        "unresolved_doi_rows": unresolved,
        "evidence_records": len(evidence_ids),
        "evidence_verification": "Model-extracted evidence is unreviewed unless independently checked; historical records are preserved, not retroactively certified.",
        "candidates": candidate_checks,
        "failed_documents": failed_docs,
        "issues": issues,
        "legacy_cached_unique_requests": db.rows("SELECT COUNT(*) count FROM llm_calls")[0][
            "count"
        ],
        "network_events_since_upgrade": db.rows(
            "SELECT status,COUNT(*) count FROM llm_request_events GROUP BY status"
        ),
        "claims": "Calculability and ML gain are not mechanism/novelty/external validation.",
    }
