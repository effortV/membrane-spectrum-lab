from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .utils import json_dumps, utc_now


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_key TEXT PRIMARY KEY,
    doi TEXT,
    title TEXT NOT NULL,
    year INTEGER,
    source TEXT NOT NULL,
    external_id TEXT,
    landing_url TEXT,
    pdf_url TEXT,
    local_path TEXT,
    sha256 TEXT,
    status TEXT NOT NULL DEFAULT 'discovered',
    reason TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(doi);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

CREATE TABLE IF NOT EXISTS literature_batches (
    batch_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    query TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    total INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS literature_batch_items (
    batch_id TEXT NOT NULL REFERENCES literature_batches(batch_id),
    doc_key TEXT NOT NULL REFERENCES documents(doc_key),
    position INTEGER NOT NULL,
    is_new INTEGER NOT NULL DEFAULT 0,
    outcome TEXT NOT NULL,
    detail TEXT,
    local_path TEXT,
    PRIMARY KEY(batch_id,doc_key)
);

CREATE TABLE IF NOT EXISTS document_references (
    dataset TEXT NOT NULL,
    reference_number INTEGER NOT NULL,
    doc_key TEXT NOT NULL REFERENCES documents(doc_key) ON DELETE CASCADE,
    source_path TEXT NOT NULL,
    PRIMARY KEY(dataset, reference_number, doc_key)
);
CREATE INDEX IF NOT EXISTS idx_document_references_lookup
ON document_references(dataset, reference_number);

CREATE TABLE IF NOT EXISTS download_queue (
    doc_key TEXT PRIMARY KEY REFERENCES documents(doc_key) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    reason TEXT,
    last_error TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    doc_key TEXT NOT NULL REFERENCES documents(doc_key) ON DELETE CASCADE,
    page INTEGER,
    kind TEXT NOT NULL,
    claim TEXT NOT NULL,
    locator TEXT,
    confidence REAL,
    model TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_doc ON evidence(doc_key);

CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    prompt_hash TEXT,
    model TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    hypothesis_id TEXT,
    run_dir TEXT NOT NULL,
    data_hash TEXT,
    split_hash TEXT,
    metrics_json TEXT NOT NULL,
    decision TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

CREATE TABLE IF NOT EXISTS llm_calls (
    request_hash TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    response_json TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_request_events (
    event_id TEXT PRIMARY KEY,
    request_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_page_runs (
    doc_key TEXT NOT NULL REFERENCES documents(doc_key) ON DELETE CASCADE,
    pdf_sha256 TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    model TEXT NOT NULL,
    page INTEGER NOT NULL,
    status TEXT NOT NULL,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(doc_key,pdf_sha256,prompt_sha256,model,page)
);

CREATE TABLE IF NOT EXISTS chat_conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    memory TEXT NOT NULL DEFAULT '',
    archived INTEGER NOT NULL DEFAULT 0 CHECK(archived IN (0,1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_turns (
    turn_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES chat_conversations(conversation_id),
    question TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK(status IN ('running','completed','failed','cancelled','timed_out','interrupted')),
    error TEXT,
    model TEXT NOT NULL,
    task_id TEXT NOT NULL,
    runtime_id TEXT NOT NULL,
    history_turns INTEGER NOT NULL DEFAULT 0,
    omitted_turns INTEGER NOT NULL DEFAULT 0,
    memory_snapshot TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_chat_turns_conversation ON chat_turns(conversation_id,created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_one_running_turn
ON chat_turns(conversation_id) WHERE status='running';
"""


class StateDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def initialize(self) -> None:
        with self.connect() as con:
            con.executescript(SCHEMA)

    def add_artifact(self, kind: str, path: Path, digest: str, metadata: dict[str, Any]) -> str:
        artifact_id = uuid.uuid4().hex
        with self.connect() as con:
            con.execute(
                "INSERT INTO artifacts(artifact_id,kind,path,sha256,metadata_json,created_at) VALUES (?,?,?,?,?,?)",
                (artifact_id, kind, str(path), digest, json_dumps(metadata), utc_now()),
            )
        return artifact_id

    def upsert_document(self, document: dict[str, Any]) -> str:
        now = utc_now()
        key = str(document["doc_key"])
        values = {
            "doc_key": key,
            "doi": document.get("doi"),
            "title": document.get("title") or "Untitled",
            "year": document.get("year"),
            "source": document.get("source") or "unknown",
            "external_id": document.get("external_id"),
            "landing_url": document.get("landing_url"),
            "pdf_url": document.get("pdf_url"),
            "local_path": document.get("local_path"),
            "sha256": document.get("sha256"),
            "status": document.get("status") or "discovered",
            "reason": document.get("reason"),
            "metadata_json": json_dumps(document.get("metadata") or {}),
            "created_at": now,
            "updated_at": now,
        }
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO documents
                (doc_key, doi, title, year, source, external_id, landing_url, pdf_url,
                 local_path, sha256, status, reason, metadata_json, created_at, updated_at)
                VALUES (:doc_key, :doi, :title, :year, :source, :external_id,
                        :landing_url, :pdf_url, :local_path, :sha256, :status,
                        :reason, :metadata_json, :created_at, :updated_at)
                ON CONFLICT(doc_key) DO UPDATE SET
                  doi=COALESCE(excluded.doi, documents.doi),
                  title=CASE WHEN excluded.title='Untitled' THEN documents.title ELSE excluded.title END,
                  year=COALESCE(excluded.year, documents.year),
                  landing_url=COALESCE(excluded.landing_url, documents.landing_url),
                  pdf_url=COALESCE(excluded.pdf_url, documents.pdf_url),
                  local_path=COALESCE(excluded.local_path, documents.local_path),
                  sha256=COALESCE(excluded.sha256, documents.sha256),
                  status=CASE WHEN documents.local_path IS NOT NULL
                              THEN documents.status ELSE excluded.status END,
                  reason=CASE WHEN documents.local_path IS NOT NULL
                              THEN documents.reason ELSE excluded.reason END,
                  metadata_json=excluded.metadata_json,
                  updated_at=excluded.updated_at
                """,
                values,
            )
        return key

    def queue_download(self, doc_key: str, reason: str | None = None) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO download_queue(doc_key, status, attempts, reason, updated_at)
                VALUES (?, 'queued', 0, ?, ?)
                ON CONFLICT(doc_key) DO UPDATE SET
                  status='queued', reason=excluded.reason, updated_at=excluded.updated_at
                """,
                (doc_key, reason, utc_now()),
            )

    def update_download(
        self, doc_key: str, status: str, error: str | None = None, increment: bool = True
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                UPDATE download_queue SET status=?,
                  attempts=attempts + ?, last_error=?, updated_at=? WHERE doc_key=?
                """,
                (status, 1 if increment else 0, error, utc_now(), doc_key),
            )
            con.execute(
                "UPDATE documents SET status=?, reason=?, updated_at=? WHERE doc_key=?",
                (status, error, utc_now(), doc_key),
            )

    def rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as con:
            return [dict(row) for row in con.execute(sql, params).fetchall()]

    def add_evidence(self, item: dict[str, Any]) -> str:
        evidence_id = item.get("evidence_id") or str(uuid.uuid4())
        with self.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO evidence
                (evidence_id, doc_key, page, kind, claim, locator, confidence,
                 model, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    item["doc_key"],
                    item.get("page"),
                    item.get("kind", "claim"),
                    item["claim"],
                    item.get("locator"),
                    item.get("confidence"),
                    item.get("model"),
                    json_dumps(item.get("payload") or {}),
                    utc_now(),
                ),
            )
        return evidence_id

    def save_hypothesis(
        self, spec: dict[str, Any], prompt_hash: str | None, model: str | None
    ) -> str:
        hypothesis_id = str(spec.get("hypothesis_id") or uuid.uuid4())
        now = utc_now()
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO hypotheses
                (hypothesis_id, name, status, spec_json, prompt_hash, model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis_id,
                    spec.get("name", "unnamed"),
                    spec.get("status", "proposed"),
                    json_dumps(spec),
                    prompt_hash,
                    model,
                    now,
                    now,
                ),
            )
        return hypothesis_id

    def cache_get(self, request_hash: str) -> dict[str, Any] | None:
        rows = self.rows(
            "SELECT response_json FROM llm_calls WHERE request_hash=?", (request_hash,)
        )
        return json.loads(rows[0]["response_json"]) if rows else None

    def cache_put(
        self,
        request_hash: str,
        model: str,
        response: dict[str, Any],
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO llm_calls
                (request_hash, model, response_json, input_tokens, output_tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request_hash,
                    model,
                    json_dumps(response),
                    input_tokens,
                    output_tokens,
                    utc_now(),
                ),
            )

    def record_llm_event(
        self,
        request_hash: str,
        model: str,
        status: str,
        usage: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        usage = usage or {}
        with self.connect() as con:
            con.execute(
                "INSERT INTO llm_request_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    request_hash,
                    model,
                    status,
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    error,
                    utc_now(),
                ),
            )

    def save_evidence_page(
        self,
        doc_key: str,
        pdf_sha256: str,
        prompt_sha256: str,
        model: str,
        page: int,
        status: str,
        evidence_count: int = 0,
        error: str | None = None,
    ) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO evidence_page_runs VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(doc_key,pdf_sha256,prompt_sha256,model,page) DO UPDATE SET status=excluded.status,evidence_count=excluded.evidence_count,error=excluded.error,updated_at=excluded.updated_at",
                (
                    doc_key,
                    pdf_sha256,
                    prompt_sha256,
                    model,
                    page,
                    status,
                    evidence_count,
                    error,
                    utc_now(),
                ),
            )
