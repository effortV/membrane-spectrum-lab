from __future__ import annotations

import csv
import ipaddress
import re
import socket
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urljoin, urlsplit

import httpx
import pymupdf

from .config import Settings
from .db import StateDB
from .library_activity import LibraryActivity
from .security import safe_error
from .utils import normalize_doi, safe_filename, sha256_file, sha256_text, utc_now


class LiteratureError(RuntimeError):
    pass


class LiteratureTransientError(LiteratureError):
    pass


def validate_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise LiteratureError(
            "Only public HTTP(S) download URLs without embedded credentials are allowed"
        )
    try:
        addresses = socket.getaddrinfo(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        )
    except OSError as exc:
        raise LiteratureTransientError("Download host cannot be resolved") from exc
    if not addresses or any(
        not ipaddress.ip_address(item[4][0].split("%", 1)[0]).is_global for item in addresses
    ):
        raise LiteratureError("Private/local network download URLs are not allowed")


def validate_downloaded_pdf(path: Path) -> None:
    with path.open("rb") as handle:
        if handle.read(4) != b"%PDF":
            raise LiteratureError("URL did not return a PDF")
        handle.seek(max(0, path.stat().st_size - 8192))
        if b"%%EOF" not in handle.read():
            raise LiteratureError("PDF appears incomplete (missing EOF); partial file retained")
    try:
        with pymupdf.open(path) as document:
            if len(document) < 1 or document.is_repaired:
                raise LiteratureError("PDF is empty or requires corruption repair")
    except (RuntimeError, ValueError) as exc:
        raise LiteratureError("Downloaded PDF is not readable") from exc


def _document_key(doi: str | None, external_id: str | None, title: str) -> str:
    if doi:
        return f"doi:{doi}"
    if external_id:
        return f"id:{external_id}"
    return f"title:{sha256_text(title.lower().strip())[:24]}"


class LiteratureService:
    def __init__(self, settings: Settings, db: StateDB, *, progress=None, check_cancelled=None):
        self.settings = settings
        self.db = db
        self.progress = progress or (lambda **_: None)
        self.check = check_cancelled or (lambda: None)
        self.activity = LibraryActivity(db)
        self.last_batch_id = None
        agent = settings.openalex_mailto or "xps-discovery-agent/local"
        self.client = httpx.Client(
            timeout=45,
            follow_redirects=False,
            headers={"User-Agent": f"XPS-Discovery-Agent/0.1 ({agent})"},
        )

    def search_openalex(self, query: str, max_results: int = 50) -> list[str]:
        if not isinstance(query, str) or not query.strip() or not 1 <= max_results <= 100:
            raise ValueError("OpenAlex requires a query and max_results between 1 and 100")
        self.last_batch_id = self.activity.start("search", query.strip())
        try:
            keys = self._search_openalex(query, max_results)
        except Exception as exc:
            from .tasks import OperationStopped

            self.activity.finish(
                self.last_batch_id,
                "cancelled" if isinstance(exc, OperationStopped) else "failed",
                safe_error(exc),
            )
            raise
        self.activity.finish(self.last_batch_id)
        return keys

    def _search_openalex(self, query: str, max_results: int = 50) -> list[str]:
        if not isinstance(query, str) or not query.strip() or not 1 <= max_results <= 100:
            raise ValueError(
                "OpenAlex requires a non-empty query and max_results between 1 and 100"
            )
        params: dict[str, Any] = {
            "search": query,
            "per-page": min(max_results, 100),
            "select": (
                "id,doi,title,publication_year,primary_location,best_oa_location,"
                "open_access,authorships,concepts"
            ),
        }
        if self.settings.openalex_api_key:
            params["api_key"] = self.settings.openalex_api_key
        if self.settings.openalex_mailto:
            params["mailto"] = self.settings.openalex_mailto
        response = None
        for attempt in range(4):
            self.check()
            self.progress(stage="检索 OpenAlex", attempt=attempt + 1, wait_limit_seconds=45)
            try:
                response = self.client.get("https://api.openalex.org/works", params=params)
                if response.status_code != 429 and response.status_code < 500:
                    break
            except httpx.HTTPError as exc:
                if attempt == 3:
                    raise LiteratureError(
                        f"OpenAlex transport failure: {type(exc).__name__}"
                    ) from None
            time.sleep(2**attempt)
        if response is None:
            raise LiteratureError("OpenAlex did not return a response")
        if response.status_code == 403:
            raise LiteratureError("OpenAlex rejected the request; check OPENALEX_API_KEY")
        if response.status_code >= 400:
            raise LiteratureError(
                f"OpenAlex HTTP {response.status_code}; no credential URL is logged"
            )
        keys: list[str] = []
        works = response.json().get("results", [])[:max_results]
        for work in works:
            self.check()
            doi = normalize_doi(work.get("doi"))
            title = work.get("title") or "Untitled"
            best = work.get("best_oa_location") or {}
            primary = work.get("primary_location") or {}
            pdf_url = best.get("pdf_url")
            landing_url = best.get("landing_page_url") or primary.get("landing_page_url")
            external_id = str(work.get("id") or "").rsplit("/", 1)[-1] or None
            doc_key = _document_key(doi, external_id, title)
            if doc_key in keys:
                continue
            is_new = not self.db.rows("SELECT doc_key FROM documents WHERE doc_key=?", (doc_key,))
            self.db.upsert_document(
                {
                    "doc_key": doc_key,
                    "doi": doi,
                    "title": title,
                    "year": work.get("publication_year"),
                    "source": "openalex",
                    "external_id": external_id,
                    "landing_url": landing_url,
                    "pdf_url": pdf_url,
                    "status": "discovered",
                    "metadata": {
                        "open_access": work.get("open_access"),
                        "primary_location": primary,
                        "authorships": work.get("authorships"),
                        "concepts": work.get("concepts"),
                        "search_query": query,
                    },
                }
            )
            local = self.db.rows("SELECT local_path FROM documents WHERE doc_key=?", (doc_key,))
            if not local or not local[0].get("local_path"):
                self.db.queue_download(doc_key, "OpenAlex discovery")
            keys.append(doc_key)
            self.activity.item(
                self.last_batch_id,
                doc_key,
                len(keys),
                is_new=is_new,
                outcome="new" if is_new else "existing",
            )
            self.progress(
                stage="保存检索结果",
                documents_total=len(works),
                documents_finished=len(keys),
                document_title=title,
            )
        return keys

    def _try_url(self, url: str, destination: Path, headers: dict[str, str] | None = None) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        validate_public_url(url)
        temporary = destination.with_suffix(f".{sha256_text(url)[:12]}.part")
        last_error = "download failed"
        for attempt in range(4):
            self.check()
            self.progress(stage="下载可访问全文", attempt=attempt + 1, wait_limit_seconds=45)
            request_headers = dict(headers or {})
            existing = temporary.stat().st_size if temporary.exists() else 0
            if existing:
                request_headers["Range"] = f"bytes={existing}-"
            try:
                current_url = url
                with ExitStack() as requests:
                    response = requests.enter_context(
                        self.client.stream("GET", current_url, headers=request_headers)
                    )
                    # Redirect responses are consumed before a new request; never forward entitlement headers across origins.
                    redirect_count = 0
                    while response.is_redirect:
                        location = response.headers.get("location")
                        if not location or redirect_count >= 8:
                            raise LiteratureError("Invalid or excessive PDF redirects")
                        next_url = urljoin(current_url, location)
                        validate_public_url(next_url)
                        if urlsplit(next_url).netloc != urlsplit(current_url).netloc:
                            request_headers = {
                                key: value
                                for key, value in request_headers.items()
                                if key.lower()
                                not in {"authorization", "x-els-apikey", "x-els-insttoken"}
                            }
                        if (
                            urlsplit(current_url).scheme == "https"
                            and urlsplit(next_url).scheme != "https"
                        ):
                            raise LiteratureError(
                                "Refusing HTTPS-to-HTTP credential/download redirect"
                            )
                        response.close()
                        response = self.client.send(
                            self.client.build_request("GET", next_url, headers=request_headers),
                            stream=True,
                        )
                        requests.callback(response.close)
                        current_url = next_url
                        redirect_count += 1
                    try:
                        self._save_pdf_response(
                            response, temporary, destination, existing, check=self.check
                        )
                    finally:
                        response.close()
                    return destination
            except LiteratureTransientError as exc:
                last_error = safe_error(exc)
                if attempt == 3:
                    break
                time.sleep(2**attempt)
            except LiteratureError:
                raise
            except httpx.HTTPError as exc:
                last_error = safe_error(exc)
                if attempt == 3:
                    break
                time.sleep(2**attempt)
        raise LiteratureTransientError(last_error)

    @staticmethod
    def _save_pdf_response(
        response: httpx.Response, temporary: Path, destination: Path, existing: int, *, check=None
    ) -> None:
        if response.status_code in {408, 429} or response.status_code >= 500:
            raise LiteratureTransientError(f"Transient PDF HTTP {response.status_code}")
        if response.status_code == 416 and existing:
            total = response.headers.get("content-range", "")
            if total != f"bytes */{existing}":
                raise LiteratureError(
                    "Range resume size mismatch; not treating partial PDF as complete"
                )
            validate_downloaded_pdf(temporary)
            temporary.replace(destination)
            return
        if response.status_code >= 400:
            raise LiteratureError(f"PDF HTTP {response.status_code}")
        append = response.status_code == 206 and existing > 0
        if response.status_code == 206:
            match = re.match(
                r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("content-range", "")
            )
            if not match or int(match.group(1)) != (existing if append else 0):
                raise LiteratureError("Invalid Content-Range; partial PDF was not appended")
        size = existing if append else 0
        limit = 150 * 1024 * 1024
        with temporary.open("ab" if append else "wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if check:
                    check()
                size += len(chunk)
                if size > limit:
                    raise LiteratureError("PDF exceeds 150 MiB download limit")
                handle.write(chunk)
        validate_downloaded_pdf(temporary)
        temporary.replace(destination)

    def _try_elsevier(self, doi: str, destination: Path) -> Path:
        if not self.settings.elsevier_api_key:
            raise LiteratureError("No Elsevier API key")
        headers = {
            "X-ELS-APIKey": self.settings.elsevier_api_key,
            "Accept": "application/pdf",
        }
        if self.settings.elsevier_insttoken:
            headers["X-ELS-Insttoken"] = self.settings.elsevier_insttoken
        url = f"https://api.elsevier.com/content/article/doi/{quote(doi, safe='')}"
        return self._try_url(url, destination, headers=headers)

    def fetch_queued(
        self, limit: int = 50, doc_keys: list[str] | None = None
    ) -> dict[str, int | str]:
        if not 1 <= limit <= 500:
            raise ValueError("Download limit must be between 1 and 500")
        self.last_batch_id = self.activity.start("fetch")
        try:
            result = self._fetch_queued(limit, doc_keys)
        except Exception as exc:
            from .tasks import OperationStopped

            self.activity.finish(
                self.last_batch_id,
                "cancelled" if isinstance(exc, OperationStopped) else "failed",
                safe_error(exc),
            )
            raise
        self.activity.finish(self.last_batch_id)
        return {**result, "batch_id": self.last_batch_id}

    def _fetch_queued(self, limit: int, doc_keys: list[str] | None) -> dict[str, int]:
        filter_sql = ""
        parameters = ()
        if doc_keys is not None:
            if not doc_keys:
                return {"downloaded": 0, "manual": 0, "failed": 0}
            if len(doc_keys) > 500:
                raise ValueError("At most 500 document keys are allowed")
            filter_sql = " AND d.doc_key IN (" + ",".join("?" for _ in doc_keys) + ")"
            parameters = tuple(doc_keys)
        rows = self.db.rows(
            """
            SELECT d.*, q.attempts FROM documents d
            JOIN download_queue q USING(doc_key)
            WHERE q.status IN ('queued', 'retry')
            """
            + filter_sql
            + """
            ORDER BY d.year DESC, d.title LIMIT ?
            """,
            parameters + (limit,),
        )
        counts = {"downloaded": 0, "manual": 0, "failed": 0}
        from .tasks import OperationStopped

        self.progress(stage="准备全文下载", documents_total=len(rows), documents_finished=0)
        for position, row in enumerate(rows, start=1):
            self.activity.item(self.last_batch_id, row["doc_key"], position)
        for index, row in enumerate(rows, start=1):
            self.check()
            self.progress(stage="获取可访问全文", document_index=index, document_title=row["title"])
            slug = safe_filename(f"{row.get('year') or 'unknown'}_{row['title']}")
            suffix = sha256_text(row["doc_key"])[:8]
            destination = self.settings.workspace_root / "library" / f"{slug}_{suffix}.pdf"
            errors: list[str] = []
            transient_failure = False
            try:
                if row.get("pdf_url"):
                    try:
                        path = self._try_url(row["pdf_url"], destination)
                    except OperationStopped:
                        raise
                    except Exception as exc:  # continue to entitlement route
                        errors.append(f"Open URL: {safe_error(exc)}")
                        transient_failure = isinstance(
                            exc, (LiteratureTransientError, httpx.TransportError)
                        )
                        if row.get("doi"):
                            path = self._try_elsevier(row["doi"], destination)
                        else:
                            raise
                elif row.get("doi"):
                    path = self._try_elsevier(row["doi"], destination)
                else:
                    raise LiteratureError("No PDF URL or DOI")
                digest = sha256_file(path)
                with self.db.connect() as con:
                    con.execute(
                        """
                        UPDATE documents SET local_path=?, sha256=?, status='downloaded',
                          reason=NULL, updated_at=? WHERE doc_key=?
                        """,
                        (str(path), digest, utc_now(), row["doc_key"]),
                    )
                self.db.update_download(row["doc_key"], "downloaded", increment=True)
                counts["downloaded"] += 1
                self.activity.item(
                    self.last_batch_id,
                    row["doc_key"],
                    index,
                    outcome="downloaded",
                    local_path=str(path),
                )
            except OperationStopped:
                raise
            except Exception as exc:
                errors.append(safe_error(exc))
                reason = "; ".join(errors)[-1000:]
                # Access failures are a normal manual-upload case, not a hidden error.
                retryable = transient_failure or isinstance(
                    exc, (LiteratureTransientError, httpx.TransportError)
                )
                self.db.update_download(
                    row["doc_key"], "retry" if retryable else "manual", reason, increment=True
                )
                counts["failed" if retryable else "manual"] += 1
                self.activity.item(
                    self.last_batch_id,
                    row["doc_key"],
                    index,
                    outcome="retry" if retryable else "manual",
                    detail=reason,
                )
            self.progress(documents_finished=index, **counts)
            time.sleep(max(0.0, self.settings.download_rate_seconds))
        self.export_manual_queue()
        return counts

    def export_manual_queue(self) -> Path:
        destination = self.settings.workspace_root / "inbox" / "missing_literature.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = self.db.rows(
            """
            SELECT d.title, d.doi, d.year, d.landing_url, d.reason, d.doc_key
            FROM documents d WHERE d.status='manual' ORDER BY d.year DESC, d.title
            """
        )
        with destination.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["title", "doi", "year", "landing_url", "reason", "doc_key"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return destination

    def index_local_pdfs(self, roots: Iterable[Path]) -> dict[str, int]:
        from .pdfs import sniff_pdf_identity

        added = 0
        duplicate = 0
        seen_hashes = {
            row["sha256"]
            for row in self.db.rows("SELECT sha256 FROM documents WHERE sha256 IS NOT NULL")
        }
        roots = list(roots)
        total = sum(1 for root in roots if root.exists() for _ in root.rglob("*.pdf"))
        self.progress(stage="索引本地 PDF", documents_total=total, documents_finished=0)
        processed = 0
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*.pdf"):
                self.check()
                processed += 1
                self.progress(
                    stage="索引本地 PDF",
                    document_index=processed,
                    document_title=path.name,
                    documents_finished=processed - 1,
                )
                digest = sha256_file(path)
                reference_dataset = None
                reference_number = None
                try:
                    relative = path.resolve().relative_to(self.settings.reference_root.resolve())
                    if relative.parts and relative.parts[0].upper() in {"NF", "RO"}:
                        reference_dataset = relative.parts[0].upper()
                        match = re.match(r"^(\d+)(?:[.\-_]|$)", path.name)
                        reference_number = int(match.group(1)) if match else None
                except ValueError:
                    pass
                if digest in seen_hashes:
                    duplicate += 1
                    existing_hash = self.db.rows(
                        "SELECT doc_key FROM documents WHERE sha256=? LIMIT 1", (digest,)
                    )
                    if existing_hash and reference_dataset and reference_number is not None:
                        with self.db.connect() as con:
                            con.execute(
                                """
                                INSERT OR IGNORE INTO document_references
                                (dataset, reference_number, doc_key, source_path) VALUES (?, ?, ?, ?)
                                """,
                                (
                                    reference_dataset,
                                    reference_number,
                                    existing_hash[0]["doc_key"],
                                    str(path.resolve()),
                                ),
                            )
                    continue
                identity = sniff_pdf_identity(path)
                doi = normalize_doi(identity.get("doi"))
                title = identity.get("title") or path.stem
                doc_key = _document_key(doi, None, title)
                # Same DOI can have two physical versions; retain provenance by hash.
                existing = self.db.rows("SELECT sha256 FROM documents WHERE doc_key=?", (doc_key,))
                if existing and existing[0].get("sha256") not in {None, digest}:
                    doc_key = f"{doc_key}:sha256:{digest[:12]}"
                self.db.upsert_document(
                    {
                        "doc_key": doc_key,
                        "doi": doi,
                        "title": title,
                        "source": "local",
                        "local_path": str(path.resolve()),
                        "sha256": digest,
                        "status": "indexed",
                        "metadata": {
                            "original_path": str(path.resolve()),
                            "reference_dataset": reference_dataset,
                            "reference_number": reference_number,
                        },
                    }
                )
                with self.db.connect() as con:
                    if reference_dataset and reference_number is not None:
                        con.execute(
                            """
                            INSERT OR IGNORE INTO document_references
                            (dataset, reference_number, doc_key, source_path) VALUES (?, ?, ?, ?)
                            """,
                            (reference_dataset, reference_number, doc_key, str(path.resolve())),
                        )
                    con.execute(
                        "UPDATE download_queue SET status='satisfied_local', updated_at=? WHERE doc_key=?",
                        (utc_now(), doc_key),
                    )
                seen_hashes.add(digest)
                added += 1
        self.progress(documents_finished=processed)
        return {"added": added, "duplicates": duplicate}
