"""Read-only, cross-session task monitoring without prompts or model answers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

from .security import safe_error


PROGRESS_FIELDS = {
    "stage",
    "document_title",
    "doc_key",
    "document_index",
    "documents_total",
    "documents_finished",
    "page",
    "page_index",
    "pages_total",
    "pages_completed",
    "pages_failed",
    "pages_skipped",
    "fits_total",
    "fits_finished",
    "outer_fold",
    "outer_total",
    "inner_fold",
    "candidate_index",
    "feature_set",
    "model",
    "attempt",
    "max_attempts",
    "network_requests",
    "wait_limit_seconds",
    "images_total",
    "parsed",
    "failed",
    "downloaded",
    "manual",
    "skipped",
    "indexed",
    "duplicates",
    "batch_evidence_saved",
    "last_error",
    "received_chunks",
    "generated_chars",
    "reasoning_chars",
    "tool_calls_count",
    "request_elapsed_seconds",
    "request_trace_id",
    "request_total_limit_seconds",
    "history_turns",
    "omitted_turns",
    "conversation_id",
}
COUNTER_FIELDS = {
    "matched",
    "existing",
    "added_or_updated",
    "saved",
    "rejected",
    "parsed",
    "failed",
    "fetched",
    "downloaded",
    "manual",
    "skipped",
    "queued",
    "indexed",
    "added",
    "duplicates",
    "documents",
    "pages_completed",
    "pages_skipped",
    "pages_failed",
    "documents_failed",
    "selected_page_count",
    "calls",
    "cache_hits",
}
STATUSES = {
    "running": "运行中",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已停止",
    "timed_out": "达到时限",
    "interrupted": "已中断（服务重启）",
}
RESULT_PAGES = {
    "search": "文献中心",
    "fetch": "文献中心",
    "index": "文献中心",
    "parse": "文献中心",
    "plan": "文献中心",
    "prepare": "文献中心",
    "extraction": "证据与智能体",
    "proposal": "证据与智能体",
    "ask": "证据与智能体",
    "relationship": "关系研究（ML）",
    "research_interpretation": "关系研究（ML）",
    "membrane": "XPS 图分析",
}


def public_progress(progress: object) -> dict[str, Any]:
    if not isinstance(progress, dict):
        return {}
    clean = {}
    for key, value in progress.items():
        if key not in PROGRESS_FIELDS:
            continue
        if isinstance(value, str):
            clean[key] = safe_error(value, limit=1200)
        elif isinstance(value, (int, float)) and math.isfinite(value):
            clean[key] = value
    return clean


def summarize_result(result: object) -> dict[str, int | float]:
    """Only aggregate counters, never questions, responses, credentials or arbitrary paths."""
    summary = {}
    if not isinstance(result, dict):
        return summary
    for key, value in result.items():
        if key in COUNTER_FIELDS and isinstance(value, (int, float)) and math.isfinite(value):
            summary[key] = value
        elif key in {"index", "text", "plan"} and isinstance(value, dict):
            summary.update(
                {f"{key}.{name}": count for name, count in summarize_result(value).items()}
            )
    return summary


def progress_measure(progress: dict[str, Any]) -> tuple[float, str] | None:
    for total_key, done_key, label in (
        ("fits_total", "fits_finished", "训练／验证"),
        ("documents_total", "documents_finished", "文献"),
        ("pages_total", "pages_completed", "当前文献已完成页"),
    ):
        total, done = progress.get(total_key), progress.get(done_key, 0)
        if isinstance(total, (int, float)) and math.isfinite(total) and total > 0:
            if not isinstance(done, (int, float)) or not math.isfinite(done):
                done = 0
            done = max(0, min(done, total))
            return done / total, f"{label} {done:g}/{total:g}"
    return None


def _duration(start: object, end: object) -> float:
    try:
        a = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        return max(0, (b - a).total_seconds())
    except (ValueError, TypeError):
        return 0


def _clean_record(raw: dict, modified: float, active_id: str | None) -> dict:
    keys = ("task_id", "label", "kind", "status", "created_at", "finished_at", "updated_at")
    record = {key: safe_error(raw.get(key) or "", limit=500) for key in keys}
    record["progress"] = public_progress(raw.get("progress"))
    record["error"] = safe_error(raw["error"]) if raw.get("error") else None
    # Re-sanitize nested counters written by the current version.
    record["result_summary"] = (
        {
            key: value
            for key, value in (raw.get("result_summary") or {}).items()
            if key.rsplit(".", 1)[-1] in COUNTER_FIELDS
            and isinstance(value, (int, float))
            and math.isfinite(value)
        }
        if isinstance(raw.get("result_summary"), dict)
        else {}
    )
    record["events"] = []
    events = raw.get("events")
    if isinstance(events, list):
        for item in events[-40:]:
            if isinstance(item, dict):
                record["events"].append(
                    {
                        "at": safe_error(item.get("at") or "", limit=100),
                        **public_progress(item),
                    }
                )
    record["is_owner"] = False
    record["cancel_requested"] = False
    if record["status"] == "running" and record["task_id"] != active_id:
        record["status"] = "interrupted"
        record["error"] = "此记录没有对应的活动后台线程；保留已完成检查点，需重新分批启动。"
    running = record["status"] == "running"
    end = datetime.now(timezone.utc).isoformat() if running else record["finished_at"]
    record["elapsed_seconds"] = _duration(record["created_at"], end)
    elapsed = raw.get("elapsed_seconds")
    if not running and isinstance(elapsed, (int, float)) and math.isfinite(elapsed):
        record["elapsed_seconds"] = max(0, elapsed)
    record["update_age_seconds"] = max(0, time.time() - modified) if running else 0
    return record


def read_task_history(root: Path, snapshot: dict | None, limit: int = 200) -> list[dict]:
    """Combine durable journals with the manager's active thread identity.

    Compatible with the previous manager API, so a new monitor need not interrupt
    an already running paid request during deployment.
    """
    limit = max(1, min(1000, int(limit)))
    active_id = snapshot["task_id"] if snapshot and snapshot["status"] == "running" else None
    records = {}
    try:
        paths = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        paths = []
    for path in paths[:limit]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not raw.get("task_id"):
                continue
            record = _clean_record(raw, path.stat().st_mtime, active_id)
            records[record["task_id"]] = record
        except (OSError, ValueError, TypeError, OverflowError):
            continue
    if snapshot:
        task_id = snapshot["task_id"]
        record = records.get(task_id) or _clean_record(snapshot, time.time(), active_id)
        for key in ("status", "is_owner", "cancel_requested", "elapsed_seconds"):
            record[key] = snapshot.get(key, record.get(key))
        if snapshot.get("is_owner"):
            record["progress"] = public_progress(snapshot.get("progress"))
            record["error"] = safe_error(snapshot["error"]) if snapshot.get("error") else None
            record["result_summary"] = (
                summarize_result(snapshot.get("result")) or record["result_summary"]
            )
        records[task_id] = record
    return sorted(
        records.values(), key=lambda r: (r["status"] == "running", r["created_at"]), reverse=True
    )[:limit]
