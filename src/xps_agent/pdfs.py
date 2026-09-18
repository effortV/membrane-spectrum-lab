from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pymupdf

from .config import Settings
from .db import StateDB
from .tasks import OperationStopped
from .security import safe_error
from .utils import json_dumps, normalize_doi, sha256_file, utc_now


RELEVANCE_TERMS = {
    "xps": 5,
    "x-ray photoelectron": 5,
    "n 1s": 5,
    "o 1s": 5,
    "binding energy": 4,
    "deconvol": 4,
    "peak fit": 4,
    "atomic concentration": 3,
    "survey spectrum": 3,
    "surface composition": 3,
    "chemical state": 3,
}

PARSER_VERSION = "2026-09-15.2"


def sniff_pdf_identity(path: Path) -> dict[str, str | None]:
    title: str | None = None
    doi: str | None = None
    try:
        with pymupdf.open(path) as document:
            title = str((document.metadata or {}).get("title") or "").strip() or None
            text = "\n".join(
                document[index].get_text("text") for index in range(min(3, len(document)))
            )
        doi = normalize_doi(text)
        if not title:
            lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
            title = lines[0][:300] if lines else None
    except Exception:
        pass
    return {"title": title, "doi": doi}


class PDFService:
    def __init__(
        self,
        settings: Settings,
        db: StateDB,
        *,
        progress: Callable[..., None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ):
        self.settings = settings
        self.db = db
        self.progress = progress or (lambda **_: None)
        self.check = check_cancelled or (lambda: None)

    def _cache_path(self, digest: str) -> Path:
        return self.settings.workspace_root / "cache" / "pdf_text" / f"{digest}.json"

    def parse_pdf(self, path: Path) -> dict[str, Any]:
        self.check()
        digest = sha256_file(path)
        cache = self._cache_path(digest)
        if cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("parser_version") == PARSER_VERSION:
                return cached
        document = pymupdf.open(path)
        pages: list[dict[str, Any]] = []
        for page_index, page in enumerate(document):
            try:
                self.check()
            except BaseException:
                document.close()
                raise
            text = page.get_text("text")
            lower = text.lower()
            score = sum(weight * lower.count(term) for term, weight in RELEVANCE_TERMS.items())
            pages.append(
                {
                    "page": page_index + 1,
                    "text": text,
                    "image_count": len(page.get_images(full=True)),
                    "relevance_score": score,
                }
            )
        payload = {
            "parser_version": PARSER_VERSION,
            "path": str(path),
            "sha256": digest,
            "page_count": len(pages),
            "metadata": dict(document.metadata or {}),
            "pages": pages,
        }
        document.close()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json_dumps(payload), encoding="utf-8")
        return payload

    def parse_registered(self, limit: int = 100) -> dict[str, int]:
        rows = self.db.rows(
            """
            SELECT doc_key, local_path FROM documents
            WHERE local_path IS NOT NULL AND status IN ('indexed','downloaded','parse_failed')
            ORDER BY updated_at LIMIT ?
            """,
            (limit,),
        )
        parsed = 0
        failed = 0
        self.progress(stage="本地读取 PDF 文字", documents_total=len(rows), documents_finished=0)
        for index, row in enumerate(rows, start=1):
            self.check()
            self.progress(
                stage="本地读取 PDF 文字",
                document_index=index,
                document_title=Path(row["local_path"]).name,
            )
            try:
                self.parse_pdf(Path(row["local_path"]))
                with self.db.connect() as con:
                    con.execute(
                        "UPDATE documents SET status='parsed', reason=NULL, updated_at=? WHERE doc_key=?",
                        (utc_now(), row["doc_key"]),
                    )
                parsed += 1
            except OperationStopped:
                raise
            except Exception as exc:
                with self.db.connect() as con:
                    con.execute(
                        "UPDATE documents SET status='parse_failed', reason=?, updated_at=? WHERE doc_key=?",
                        (safe_error(exc), utc_now(), row["doc_key"]),
                    )
                failed += 1
            self.progress(documents_finished=index, parsed=parsed, failed=failed)
        return {"parsed": parsed, "failed": failed}

    def evidence_plan(self, limit: int = 1000) -> dict[str, Any]:
        import pandas as pd

        rows = self.db.rows(
            """
            SELECT doc_key, title, doi, local_path FROM documents
            WHERE status IN ('parsed','evidence_failed') AND local_path IS NOT NULL
            ORDER BY title LIMIT ?
            """,
            (limit,),
        )
        records: list[dict[str, Any]] = []
        failed = 0
        self.progress(stage="筛选 XPS 图页", documents_total=len(rows), documents_finished=0)
        for index, row in enumerate(rows, start=1):
            self.check()
            self.progress(stage="筛选 XPS 图页", document_index=index, document_title=row["title"])
            try:
                payload = self.parse_pdf(Path(row["local_path"]))
            except OperationStopped:
                raise
            except Exception:
                failed += 1
                self.progress(documents_finished=index, failed=failed)
                continue
            selected = self.select_relevant_pages(payload)
            records.append(
                {
                    "doc_key": row["doc_key"],
                    "title": row["title"],
                    "doi": row.get("doi"),
                    "pdf_pages": payload["page_count"],
                    "selected_pages": ",".join(map(str, selected)),
                    "selected_page_count": len(selected),
                }
            )
            self.progress(documents_finished=index)
        destination = self.settings.workspace_root / "state" / "evidence_plan.csv"
        pd.DataFrame(records).to_csv(destination, index=False, encoding="utf-8-sig")
        return {
            "documents": len(records),
            "failed": failed,
            "selected_pages": sum(item["selected_page_count"] for item in records),
            "page_cap_per_pdf": self.settings.vision_max_pages_per_pdf,
            "vision_model": self.settings.vision_model,
            "manifest": str(destination),
            "cost_note": "Only selected pages are sent; cached page/model hashes prevent repeat billing.",
        }

    def select_relevant_pages(self, payload: dict[str, Any], limit: int | None = None) -> list[int]:
        max_pages = limit or self.settings.vision_max_pages_per_pdf
        pages = payload.get("pages", [])
        ranked = sorted(
            pages,
            key=lambda page: (
                int(page.get("relevance_score", 0)),
                int(page.get("image_count", 0) > 0),
            ),
            reverse=True,
        )
        selected = [int(item["page"]) for item in ranked if item.get("relevance_score", 0) >= 4]
        return selected[:max_pages]

    def render_pages(self, path: Path, pages: list[int], digest: str) -> list[Path]:
        document = pymupdf.open(path)
        output_dir = self.settings.workspace_root / "cache" / "pages" / digest
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for page_number in pages:
            output = output_dir / f"page_{page_number:04d}.png"
            if not output.exists():
                page = document[page_number - 1]
                scale = min(1.7, 2400 / max(page.rect.width, page.rect.height, 1.0))
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                pixmap.save(output)
            outputs.append(output)
        document.close()
        return outputs
