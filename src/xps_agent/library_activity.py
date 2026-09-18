"""Durable literature acquisition receipts. Never guess pre-upgrade new counts."""

from __future__ import annotations

import json
import uuid
from .db import StateDB
from .security import safe_error
from .utils import utc_now


class LibraryActivity:
    def __init__(self, db: StateDB):
        self.db = db

    def start(self, kind: str, query: str = "") -> str:
        key = uuid.uuid4().hex
        with self.db.connect() as con:
            con.execute(
                "INSERT INTO literature_batches(batch_id,kind,query,created_at) VALUES (?,?,?,?)",
                (key, kind, safe_error(query) if query else "", utc_now()),
            )
        return key

    def item(
        self,
        batch_id,
        doc_key,
        position,
        *,
        is_new=False,
        outcome="pending",
        detail=None,
        local_path=None,
    ):
        with self.db.connect() as con:
            con.execute(
                """INSERT INTO literature_batch_items
                   (batch_id,doc_key,position,is_new,outcome,detail,local_path)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(batch_id,doc_key) DO UPDATE SET
                   outcome=excluded.outcome,detail=excluded.detail,local_path=excluded.local_path""",
                (
                    batch_id,
                    doc_key,
                    position,
                    int(is_new),
                    outcome,
                    safe_error(detail) if detail else None,
                    local_path,
                ),
            )

    def finish(self, batch_id, status="completed", error=None):
        with self.db.connect() as con:
            con.execute(
                """UPDATE literature_batches SET status=?,error=?,finished_at=?,
                   total=(SELECT COUNT(*) FROM literature_batch_items WHERE batch_id=?)
                   WHERE batch_id=?""",
                (status, safe_error(error) if error else None, utc_now(), batch_id, batch_id),
            )

    def batches(self, kind="search", limit=100):
        return self.db.rows(
            "SELECT * FROM literature_batches WHERE kind=? ORDER BY created_at DESC,rowid DESC LIMIT ?",
            (kind, limit),
        )

    def items(self, batch_id):
        return self.db.rows(
            """SELECT i.*,d.title,d.doi,d.year,d.source,d.status,d.landing_url,d.local_path AS current_path,
                      COALESCE(q.status,'') AS download_status,COALESCE(q.last_error,d.reason) AS reason
               FROM literature_batch_items i JOIN documents d USING(doc_key)
               LEFT JOIN download_queue q USING(doc_key)
               WHERE i.batch_id=? ORDER BY i.position""",
            (batch_id,),
        )

    def summary(self, batch_id):
        rows = self.items(batch_id)
        return {
            "batch_id": batch_id,
            "matched": len(rows),
            "added": sum(bool(row["is_new"]) for row in rows),
            "existing": sum(not row["is_new"] for row in rows),
        }

    def documents(self):
        return self.db.rows(
            """SELECT d.*,COALESCE(q.status,'') AS download_status,
               COALESCE(q.last_error,d.reason) AS download_reason
               FROM documents d LEFT JOIN download_queue q USING(doc_key)
               ORDER BY d.created_at DESC,d.title"""
        )

    def legacy_search_items(self):
        rows = []
        for row in self.documents():
            try:
                metadata = json.loads(row["metadata_json"])
            except (TypeError, ValueError):
                metadata = {}
            if row["source"] == "openalex" or metadata.get("search_query"):
                rows.append(
                    {
                        **row,
                        "is_new": None,
                        "outcome": "legacy",
                        "current_path": row.get("local_path"),
                        "reason": row.get("download_reason"),
                        "search_query": metadata.get("search_query", ""),
                    }
                )
        return rows
