"""Authenticated FastAPI embedded in the LOCAL dashboard process.

Do not run another uvicorn worker against the same database: the local UI and
this API intentionally share paid gate, BackgroundTasks and conversation runtime.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import socket
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import __version__
from .backend_jobs import JOB_SPECS, PAID_JOBS, build_worker, confined_file
from .config import Settings
from .conversations import ConversationStore, start_conversation_task, validate_text
from .library_activity import LibraryActivity
from .presentation import normalize_math_markdown
from .security import safe_error
from .task_monitor import public_progress, read_task_history
from .utils import utc_now

MAX_BODY = 32 * 1024 * 1024
OWNER_PATTERN = re.compile(r"^[a-f0-9]{32}$")


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class JobBody(StrictBody):
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    kind: str = Field(max_length=40)
    parameters: dict[str, Any] = Field(default_factory=dict)
    confirmed_paid: bool = False


class ChatBody(StrictBody):
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    question: str = Field(min_length=1, max_length=6000)
    confirmed_paid: bool = False


class ConversationBody(StrictBody):
    title: str = Field(default="新研究会话", min_length=1, max_length=120)
    memory: str = Field(default="", max_length=6000)


class UploadBody(StrictBody):
    name: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1, max_length=28 * 1024 * 1024)
    kind: str = Field(pattern=r"^(pdf|xps_image)$")


def public(value):
    """Scrub known credentials without truncating scientific answers."""
    if isinstance(value, str):
        return safe_error(value, limit=len(value))
    if isinstance(value, dict):
        return {key: public(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public(item) for item in value]
    return value


def create_app(settings: Settings, db, manager, token: str) -> FastAPI:
    if not isinstance(token, str) or len(token) < 32 or any(char.isspace() for char in token):
        raise ValueError("A strong backend token is required.")
    app = FastAPI(
        title="Membrane research backend", docs_url=None, redoc_url=None, openapi_url=None
    )
    request_lock = threading.RLock()
    with db.connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS backend_requests (
            request_id TEXT PRIMARY KEY,owner TEXT NOT NULL,payload_hash TEXT NOT NULL,
            status TEXT NOT NULL,task_id TEXT,error TEXT,created_at TEXT NOT NULL)""")

    def current_settings():
        latest = Settings.load(settings.project_root)
        if latest.database_path.resolve() != db.path.resolve():
            raise RuntimeError("数据目录发生变化，请在所有任务结束后重启本机服务。")
        return latest

    @app.middleware("http")
    async def guard(request: Request, call_next):
        provided = request.headers.get("authorization", "")
        if not hmac.compare_digest(provided.encode(), ("Bearer " + token).encode()):
            return JSONResponse({"detail": "接口凭据无效。"}, status_code=401)
        owner = request.headers.get("x-xps-owner", "")
        if not OWNER_PATTERN.fullmatch(owner):
            return JSONResponse({"detail": "会话标识无效。"}, status_code=400)
        request.state.owner = owner
        if (
            settings.workspace_root / "state" / "migration_hold.json"
        ).exists() and request.url.path != "/health":
            return JSONResponse({"detail": "服务器正在维护，未启动新任务。"}, status_code=503)
        if request.method in {"POST", "PUT", "PATCH"}:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_BODY:
                    return JSONResponse({"detail": "上传内容超过上限。"}, status_code=413)
            request._body = bytes(body)
        try:
            response = await call_next(request)
        except Exception:
            return JSONResponse(
                {"detail": "服务器处理失败，请查看后台任务状态；未自动重试。"}, status_code=500
            )
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse({"detail": "请求字段或格式不正确。"}, status_code=422)

    @app.exception_handler(ValueError)
    async def input_error(request, error):
        return JSONResponse({"detail": safe_error(error)}, status_code=400)

    def submit(owner, request_id, payload, starter):
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
        ).hexdigest()
        with request_lock:
            existing = db.rows("SELECT * FROM backend_requests WHERE request_id=?", (request_id,))
            if existing:
                row = existing[0]
                if row["owner"] != owner or row["payload_hash"] != fingerprint:
                    raise HTTPException(409, "请求编号已用于其他操作。")
                return {
                    "request_id": request_id,
                    "task_id": row["task_id"],
                    "status": row["status"],
                    "error": row["error"],
                    "duplicate": True,
                }
            with db.connect() as con:
                con.execute(
                    "INSERT INTO backend_requests(request_id,owner,payload_hash,status,created_at) VALUES (?,?,?,'reserved',?)",
                    (request_id, owner, fingerprint, utc_now()),
                )
            try:
                task_id = starter()
            except Exception as error:
                with db.connect() as con:
                    con.execute(
                        "UPDATE backend_requests SET status='failed',error=? WHERE request_id=?",
                        (safe_error(error), request_id),
                    )
                raise HTTPException(409, safe_error(error)) from None
            with db.connect() as con:
                con.execute(
                    "UPDATE backend_requests SET status='submitted',task_id=? WHERE request_id=?",
                    (task_id, request_id),
                )
            return {
                "request_id": request_id,
                "task_id": task_id,
                "status": "submitted",
                "duplicate": False,
            }

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "database_location": "server_only"}

    @app.get("/v1/overview")
    def overview():
        profile_path = settings.workspace_root / "canonical" / "data_profile.json"
        profile = (
            json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.is_file() else {}
        )
        counts = {
            name: db.rows(f"SELECT COUNT(*) AS n FROM {name}")[0]["n"]
            for name in ("documents", "evidence", "hypotheses", "experiments", "chat_conversations")
        }
        return public(
            {"counts": counts, "profile": profile, "settings": current_settings().public_summary()}
        )

    @app.get("/v1/library")
    def library():
        rows = LibraryActivity(db).documents()
        fields = (
            "doc_key",
            "doi",
            "title",
            "year",
            "source",
            "status",
            "download_status",
            "download_reason",
            "landing_url",
        )
        return public(
            [
                {**{key: row.get(key) for key in fields}, "has_pdf": bool(row.get("local_path"))}
                for row in rows
            ]
        )

    @app.get("/v1/library/batches")
    def batches(kind: str = Query("search", pattern="^(search|download)$")):
        return public(LibraryActivity(db).batches(kind))

    @app.get("/v1/library/batches/{batch_id}")
    def batch(batch_id: str):
        activity = LibraryActivity(db)
        return public({"summary": activity.summary(batch_id), "items": activity.items(batch_id)})

    @app.get("/v1/library/pdf")
    def pdf(doc_key: str):
        rows = db.rows("SELECT local_path FROM documents WHERE doc_key=?", (doc_key,))
        if not rows or not rows[0]["local_path"]:
            raise HTTPException(404, "尚无可读取的 PDF。")
        path = confined_file(
            rows[0]["local_path"],
            [
                settings.reference_root,
                settings.workspace_root / "library",
                settings.workspace_root / "inbox",
                (settings.storage_root or settings.workspace_root) / "raw" / "literature",
            ],
            ".pdf",
        )
        with path.open("rb") as handle:
            if not handle.read(8).startswith(b"%PDF-"):
                raise ValueError("文件不是 PDF。")
        return FileResponse(path, media_type="application/pdf", filename="reference.pdf")

    @app.get("/v1/evidence")
    def evidence(
        limit: int = Query(100, ge=1, le=300),
        offset: int = Query(0, ge=0),
        query: str = Query("", max_length=500),
    ):
        return public(
            db.rows(
                "SELECT e.evidence_id,e.doc_key,e.page,e.kind,e.claim,e.locator,e.confidence,d.title,d.doi FROM evidence e JOIN documents d USING(doc_key) WHERE e.claim LIKE ? ORDER BY e.created_at DESC LIMIT ? OFFSET ?",
                ("%" + query + "%", limit, offset),
            )
        )

    @app.get("/v1/hypotheses")
    def hypotheses():
        rows = db.rows(
            "SELECT hypothesis_id,name,status,spec_json,created_at FROM hypotheses ORDER BY created_at DESC"
        )
        return public([{**row, "spec": json.loads(row.pop("spec_json"))} for row in rows])

    @app.get("/v1/conversations")
    def conversations(archived: bool = False):
        return public(ConversationStore(db).list(archived=archived))

    @app.post("/v1/conversations")
    def new_conversation(body: ConversationBody):
        store = ConversationStore(db)
        key = store.create(body.title)
        store.update(key, title=body.title, memory=body.memory)
        return {"conversation_id": key}

    @app.get("/v1/conversations/{key}")
    def conversation(key: str):
        store = ConversationStore(db)
        store.recover(settings.workspace_root / "state" / "tasks")
        exported = store.export(key)
        for turn in exported["turns"]:
            turn["answer"] = normalize_math_markdown(turn["answer"])
        return public(exported)

    @app.patch("/v1/conversations/{key}")
    def update_conversation(key: str, body: ConversationBody):
        ConversationStore(db).update(key, title=body.title, memory=body.memory)
        return {"updated": True}

    @app.get("/v1/conversations/{key}/markdown")
    def conversation_markdown(key: str):
        return Response(
            public(ConversationStore(db).markdown(key)), media_type="text/markdown; charset=utf-8"
        )

    @app.post("/v1/conversations/{key}/ask", status_code=202)
    def ask(key: str, body: ChatBody, request: Request):
        if not body.confirmed_paid:
            raise HTTPException(400, "请先确认本次模型调用可能计费。")
        question = validate_text(body.question, maximum=6000, label="问题")
        latest = current_settings()
        if not latest.siliconflow_api_key:
            raise HTTPException(409, "请先在服务器本机页面配置模型密钥。")
        return submit(
            request.state.owner,
            body.request_id,
            {"key": key, "question": question},
            lambda: start_conversation_task(
                latest, db, manager, request.state.owner, key, question
            ),
        )

    @app.post("/v1/jobs", status_code=202)
    def start_job(body: JobBody, request: Request):
        if body.kind not in JOB_SPECS:
            raise HTTPException(400, "不支持该任务。")
        schema, label = JOB_SPECS[body.kind]
        try:
            params = schema.model_validate(body.parameters)
        except ValidationError:
            raise HTTPException(422, "任务参数不正确。") from None
        latest = current_settings()
        if body.kind in PAID_JOBS:
            if not body.confirmed_paid:
                raise HTTPException(400, "请先确认本次模型调用可能计费。")
            if not latest.siliconflow_api_key:
                raise HTTPException(409, "请先在服务器本机页面配置模型密钥。")
        return submit(
            request.state.owner,
            body.request_id,
            body.model_dump(),
            lambda: manager.start(
                request.state.owner,
                label,
                body.kind,
                build_worker(latest, db, body.kind, params),
                timeout=latest.action_timeout_seconds,
                journal_root=latest.workspace_root / "state" / "tasks",
            ),
        )

    @app.get("/v1/tasks")
    def tasks(request: Request):
        snapshot = manager.snapshot(request.state.owner)
        if snapshot:
            snapshot["progress"] = public_progress(snapshot["progress"])
        return public(
            {
                "current": snapshot,
                "history": read_task_history(
                    settings.workspace_root / "state" / "tasks", snapshot, 100
                ),
            }
        )

    @app.post("/v1/tasks/{task_id}/cancel")
    def cancel(task_id: str, request: Request):
        manager.cancel(task_id, request.state.owner)
        return {"cancel_requested": True}

    @app.get("/v1/requests/{request_id}")
    def request_status(request_id: str, request: Request):
        rows = db.rows(
            "SELECT request_id,status,task_id,error FROM backend_requests WHERE request_id=? AND owner=?",
            (request_id, request.state.owner),
        )
        if not rows:
            raise HTTPException(404, "没有此提交记录。")
        return public(rows[0])

    def frame():
        import pandas as pd
        from .research import current_research_table

        path = current_research_table(settings)
        if not path.is_file():
            raise HTTPException(409, "请先审计 NF / RO 数据。")
        return pd.read_csv(path, low_memory=False)

    @app.get("/v1/research/catalog")
    def catalog():
        from .research import field_catalog, load_role_overrides, TASKS, ROLE_LABELS

        data = frame()
        symbols = [
            json.loads(row["spec_json"]).get("symbol")
            for row in db.rows("SELECT spec_json FROM hypotheses WHERE status='calculable'")
        ]
        return public(
            {
                "columns": json.loads(
                    field_catalog(data, load_role_overrides(settings), symbols).to_json(
                        orient="records", force_ascii=False
                    )
                ),
                "relationships": TASKS,
                "roles": ROLE_LABELS,
                "datasets": sorted(data["dataset"].dropna().unique().tolist()),
            }
        )

    @app.get("/v1/data")
    def data(
        limit: int = Query(100, ge=1, le=300),
        offset: int = Query(0, ge=0),
        dataset: str = Query("", pattern="^(|NF|RO)$"),
    ):
        values = frame()
        if dataset:
            values = values[values["dataset"] == dataset]
        return public(
            {
                "total": len(values),
                "columns": values.columns.tolist(),
                "rows": json.loads(
                    values.iloc[offset : offset + limit].to_json(
                        orient="records", force_ascii=False
                    )
                ),
            }
        )

    @app.get("/v1/spectrum")
    def spectrum(record_id: str = Query(min_length=1, max_length=500)):
        from .data import DataAuditor

        values = frame()
        selected = values[values["record_id"].astype(str) == record_id]
        if len(selected) != 1:
            raise HTTPException(404, "没有唯一对应的谱图样品。")
        output = {}
        for element in ("N", "O"):
            source = selected.iloc[0].get(element + "__spectrum_source_file")
            if isinstance(source, str) and source:
                suffix = Path(source).suffix.lower()
                if suffix not in {".csv", ".xls", ".xlsx"}:
                    raise ValueError("谱图文件格式不支持。")
                path = confined_file(
                    source,
                    [
                        settings.legacy_root,
                        settings.data_root,
                        settings.workspace_root / "canonical",
                        (settings.storage_root or settings.workspace_root) / "raw",
                    ],
                    suffix,
                )
                energy, intensity = DataAuditor._load_spectrum_xy(path)
                output[element] = {
                    "energy_eV": energy.tolist(),
                    "normalized_intensity": intensity.tolist(),
                }
        return public(output)

    @app.get("/v1/analyses")
    def analyses():
        return public(
            db.rows(
                "SELECT artifact_id,metadata_json,created_at FROM artifacts WHERE kind='membrane_analysis' ORDER BY created_at DESC LIMIT 50"
            )
        )

    @app.get("/v1/analyses/{key}")
    def analysis(key: str):
        rows = db.rows(
            "SELECT path FROM artifacts WHERE artifact_id=? AND kind='membrane_analysis'", (key,)
        )
        if not rows:
            raise HTTPException(404, "没有此图谱分析结果。")
        path = confined_file(rows[0]["path"], [settings.workspace_root / "runs"], ".json")
        return public(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/v1/experiments")
    def experiments():
        return public(
            db.rows(
                "SELECT experiment_id,decision,created_at FROM experiments ORDER BY created_at DESC LIMIT 100"
            )
        )

    @app.get("/v1/experiments/{key}")
    def experiment(key: str):
        rows = db.rows("SELECT metrics_json FROM experiments WHERE experiment_id=?", (key,))
        if not rows:
            raise HTTPException(404, "没有此实验。")
        return public(json.loads(rows[0]["metrics_json"]))

    @app.post("/v1/uploads")
    def upload(body: UploadBody):
        try:
            content = base64.b64decode(body.content_base64, validate=True)
        except (ValueError, TypeError):
            raise ValueError("上传内容编码无效。") from None
        if not content or len(content) > 20 * 1024 * 1024:
            raise ValueError("文件必须小于 20 MiB。")
        if body.kind == "xps_image":
            from .membrane import save_xps_image

            record = save_xps_image(current_settings(), body.name, content)
            key = db.add_artifact(
                "xps_uploaded_image", Path(record["original"]), record["sha256"], record
            )
            return {
                "artifact_id": key,
                "name": record["name"],
                "sha256": record["sha256"],
                "width": record["width"],
                "height": record["height"],
            }
        if not content.startswith(b"%PDF-"):
            raise ValueError("请上传真实的 PDF。")
        digest = hashlib.sha256(content).hexdigest()
        target = settings.workspace_root / "inbox" / f"manual_{digest}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as handle:
                handle.write(content)
        key = db.upsert_document(
            {
                "doc_key": "upload:" + digest,
                "title": Path(body.name.replace("\\", "/")).name,
                "source": "uploaded_manual",
                "status": "indexed",
                "local_path": str(target),
                "sha256": digest,
            }
        )
        return {"doc_key": key, "sha256": digest, "status": "indexed"}

    return app


@dataclass
class EmbeddedAPI:
    server: Any
    thread: threading.Thread


def _open_listener():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 8771))
    listener.listen(128)
    return listener


def start_embedded_api(settings, db, manager) -> EmbeddedAPI | None:
    """Opt-in using a restricted server-side token file; never bind the LAN."""
    token_path = Path(
        os.getenv("XPS_BACKEND_TOKEN_FILE")
        or settings.workspace_root / "state" / "connections" / "backend_token.txt"
    )
    if not token_path.is_file():
        return None
    token = token_path.read_text(encoding="utf-8").strip()
    import uvicorn

    app = create_app(settings, db, manager, token)
    listener = _open_listener()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=8771, access_log=False, log_level="error", workers=1
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True, name="xps-fastapi"
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        listener.close()
        raise RuntimeError("服务器接口启动失败，未切换数据库或自动重试。")
    return EmbeddedAPI(server, thread)
